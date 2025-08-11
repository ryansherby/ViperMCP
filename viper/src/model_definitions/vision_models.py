import warnings
import re
import contextlib
import os
import timeit
from typing import List, Tuple, Union
import warnings
import os
from collections import Counter
from itertools import chain
from joblib import Memory

import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F




from .base_model import BaseModel
from .openai_model import OpenAIModel

from viper.configs import config




class Midas(BaseModel):
    name = 'midas'

    def __init__(self, gpu_number=0):
        super().__init__(gpu_number)
        
        from transformers import DPTImageProcessor, DPTForDepthEstimation
        
        with self.hp('Midas'):
            

            warnings.simplefilter("ignore")

            image_processor = DPTImageProcessor.from_pretrained(config.depth_estimation.path)
            model = DPTForDepthEstimation.from_pretrained(
                config.depth_estimation.path if config.use_local_models else config.depth_estimation.model,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16 if self.dev == f'cuda:{gpu_number}' else torch.float32
            )
        self.model = model.to(self.dev)
        self.image_processor = image_processor

    @torch.no_grad()
    def forward(self, image):
        """Estimate depth map"""
        
        
        image_numpy = image.cpu().permute(1, 2, 0).numpy() * 255
        input_batch = self.image_processor(image_numpy, return_tensors='pt').to(self.dev)
        prediction = self.model(**input_batch).predicted_depth
        # Resize to original size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_numpy.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        # We compute the inverse because the model returns inverse depth
        to_return = 1 / prediction
        to_return = to_return.cpu()
        return to_return  # To save: plt.imsave(path_save, prediction.cpu().numpy())
"""    
class BLIP(BaseModel):
    name = 'blip'
    to_batch = False
    max_batch_size = 32
    seconds_collect_data = 0.2  # The queue has additionally the time it is executing the previous forward pass

    def __init__(self, gpu_number=0, half_precision=config.object_captioning.half_precision):
        super().__init__(gpu_number)

        # from lavis.models import load_model_and_preprocess
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        # https://huggingface.co/models?sort=downloads&search=Salesforce%2Fblip2-

        with warnings.catch_warnings(), self.hp("BLIP"):
            max_memory = {gpu_number: torch.cuda.mem_get_info(self.dev)[0] if self.dev == f'cuda:{gpu_number}' else None}
            
            self.half_precision = half_precision if ("cuda" in self.dev or "mps" in self.dev) else False

            self.processor = Blip2Processor.from_pretrained(config.object_captioning.path if config.use_local_models else config.object_captioning.model,
                                                            torch_dtype=torch.float16 if self.half_precision else "auto",)
            # Device_map must be sequential for manual GPU selection
            try:
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    config.object_captioning.path if config.use_local_models else config.object_captioning.model,
                    torch_dtype=torch.float16 if self.half_precision else "auto",
                    device_map="sequential", max_memory=max_memory if self.dev == f'cuda:{gpu_number}' else None,
                ).to(self.dev)
            except Exception as e:
                # Clarify error message. The problem is that it tries to load part of the model to disk.
                if "had weights offloaded to the disk" in e.args[0]:
                    extra_text = ' You may want to consider setting half_precision to True.' if self.half_precision else ''
                    raise MemoryError(f"Not enough GPU memory in GPU {self.dev} to load the model.{extra_text}")
                else:
                    raise e

        self.qa_prompt = "Question: {} Short Answer:"
        self.caption_prompt = "a photo of"
        self.max_words = 50

    @torch.no_grad()
    def caption(self, image, prompt=None):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.dev, torch.float16)
        generated_ids = self.model.generate(**inputs, length_penalty=1., num_beams=5, max_length=30, min_length=1,
                                            do_sample=False, top_p=0.9, repetition_penalty=1.0,
                                            num_return_sequences=1, temperature=1)
        generated_text = [cap.strip() for cap in
                          self.processor.batch_decode(generated_ids, skip_special_tokens=True)]
        return generated_text

    def pre_question(self, question):
        # from LAVIS blip_processors
        question = re.sub(
            r"([.!\"()*#:;~])",
            "",
            question.lower(),
        )
        question = question.rstrip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > self.max_words:
            question = " ".join(question_words[: self.max_words])

        return question

    @torch.no_grad()
    def qa(self, image, question):
        inputs = self.processor(images=image, text=question, return_tensors="pt", padding="longest").to(self.dev)
        if self.half_precision:
            inputs['pixel_values'] = inputs['pixel_values'].half()
        generated_ids = self.model.generate(**inputs, length_penalty=-1, num_beams=5, max_length=10, min_length=1,
                                            do_sample=False, top_p=0.9, repetition_penalty=1.0,
                                            num_return_sequences=1, temperature=1)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_text

    def forward(self, image, question=None, task='caption'):
        if not self.to_batch:
            image, question, task = [image], [question], [task]

        if len(image) > 0 and 'float' in str(image[0].dtype) and image[0].max() <= 1:
            image = [im * 255 for im in image]

        # Separate into qa and caption batches.
        prompts_qa = [self.qa_prompt.format(self.pre_question(q)) for q, t in zip(question, task) if t == 'qa']
        
        images_qa = [im for i, im in enumerate(image) if task[i] == 'qa']
        images_caption = [im for i, im in enumerate(image) if task[i] == 'caption']

        with torch.device(self.dev):
            response_qa = self.qa(images_qa, prompts_qa) if len(images_qa) > 0 else []
            response_caption = self.caption(images_caption) if len(images_caption) > 0 else []

        response = []
        for t in task:
            if t == 'qa':
                response.append(response_qa.pop(0))
            else:
                response.append(response_caption.pop(0))

        if not self.to_batch:
            response = response[0].strip()
        return response"""
    
"""
class GLIP(BaseModel):
    name = 'glip'

    def __init__(self, model_size='large', gpu_number=0,*args):
        BaseModel.__init__(self, gpu_number)

        with contextlib.redirect_stderr(open(os.devnull, "w")):  # Do not print nltk_data messages when importing
            from viper.src.base_models.maskrcnn_benchmark.engine.predictor_glip import GLIPDemo, to_image_list, create_positive_map, \
                create_positive_map_label_to_token_from_positive_map
                
        if not config.use_local_models:
            warnings.warn("GLIP is not available through HuggingFace API. Attempting to load from local path.")

        config_file = config.object_detection.path+"/configs/glip_Swin_L.yaml"
        weight_file = config.object_detection.path+"/checkpoints/glip_large_model.pth"

        class CustomGLIPDemo(GLIPDemo):

            def __init__(self, dev, hp, *args_demo):

                kwargs = {
                    'min_image_size': 800,
                    'confidence_threshold': config.object_detection.detection_threshold,
                    'show_mask_heatmaps': False
                }

                self.dev = dev
                self.hp = hp

                from viper.src.base_models.maskrcnn_benchmark.config import cfg

                # manual override some options
                cfg.local_rank = 0
                cfg.num_gpus = 1
                cfg.merge_from_file(config_file)
                cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
                cfg.merge_from_list(["MODEL.DEVICE", self.dev])

                with self.hp("GLIP"), torch.cuda.device(self.dev):
                    from transformers.utils import logging
                    logging.set_verbosity_error()
                    GLIPDemo.__init__(self, cfg, *args_demo, **kwargs)
                if self.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
                    plus = 1
                else:
                    plus = 0
                self.plus = plus
                self.color = 255

            @torch.no_grad()
            def compute_prediction(self, original_image, original_caption, custom_entity=None):
                image = self.transforms(original_image)
                # image = [image, image.permute(0, 2, 1)]
                image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
                image_list = image_list.to(self.dev)
                # caption
                if isinstance(original_caption, list):

                    if len(original_caption) > 40:
                        all_predictions = None
                        for loop_num, i in enumerate(range(0, len(original_caption), 40)):
                            list_step = original_caption[i:i + 40]
                            prediction_step = self.compute_prediction(original_image, list_step, custom_entity=None)
                            if all_predictions is None:
                                all_predictions = prediction_step
                            else:
                                # Aggregate predictions
                                all_predictions.bbox = torch.cat((all_predictions.bbox, prediction_step.bbox), dim=0)
                                for k in all_predictions.extra_fields:
                                    all_predictions.extra_fields[k] = \
                                        torch.cat((all_predictions.extra_fields[k],
                                                   prediction_step.extra_fields[k] + loop_num), dim=0)
                        return all_predictions

                    # we directly provided a list of category names
                    caption_string = ""
                    tokens_positive = []
                    seperation_tokens = " . "
                    for word in original_caption:
                        tokens_positive.append([len(caption_string), len(caption_string) + len(word)])
                        caption_string += word
                        caption_string += seperation_tokens

                    tokenized = self.tokenizer([caption_string], return_tensors="pt")
                    # tokens_positive = [tokens_positive]  # This was wrong
                    tokens_positive = [[v] for v in tokens_positive]

                    original_caption = caption_string
                    # print(tokens_positive)
                else:
                    tokenized = self.tokenizer([original_caption], return_tensors="pt")
                    if custom_entity is None:
                        tokens_positive = self.run_ner(original_caption)
                    # print(tokens_positive)
                # process positive map
                positive_map = create_positive_map(tokenized, tokens_positive)

                positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map,
                                                                                                   plus=self.plus)
                self.positive_map_label_to_token = positive_map_label_to_token
                tic = timeit.time.perf_counter()

                # compute predictions
                with self.hp():  # Hide some deprecated notices
                    predictions = self.model(image_list, captions=[original_caption],
                                             positive_map=positive_map_label_to_token)
                predictions = [o.to(self.cpu_device) for o in predictions]
                # print("inference time per image: {}".format(timeit.time.perf_counter() - tic))

                # always single image is passed at a time
                prediction = predictions[0]

                # reshape prediction (a BoxList) into the original image size
                height, width = original_image.shape[-2:]
                # if self.tensor_inputs:
                # else:
                #     height, width = original_image.shape[:-1]
                prediction = prediction.resize((width, height))

                if prediction.has_field("mask"):
                    # if we have masks, paste the masks in the right position
                    # in the image, as defined by the bounding boxes
                    masks = prediction.get_field("mask")
                    # always single image is passed at a time
                    masks = self.masker([masks], [prediction])[0]
                    prediction.add_field("mask", masks)

                return prediction

            @staticmethod
            def to_left_right_upper_lower(bboxes):
                return [(bbox[1], bbox[3], bbox[0], bbox[2]) for bbox in bboxes]

            @staticmethod
            def to_xmin_ymin_xmax_ymax(bboxes):
                # invert the previous method
                return [(bbox[2], bbox[0], bbox[3], bbox[1]) for bbox in bboxes]

            @staticmethod
            def prepare_image(image):
                image = image[[2, 1, 0]]  # convert to bgr for opencv-format for glip
                return image

            @torch.no_grad()
            def forward(self, image: torch.Tensor, obj: Union[str, list], return_labels: bool = False,
                        confidence_threshold=None):

                if confidence_threshold is not None:
                    original_confidence_threshold = self.confidence_threshold
                    self.confidence_threshold = confidence_threshold

                # if isinstance(object, list):
                #     object = ' . '.join(object) + ' .' # add separation tokens
                image = self.prepare_image(image)

                # Avoid the resizing creating a huge image in a pathological case
                ratio = image.shape[1] / image.shape[2]
                ratio = max(ratio, 1 / ratio)
                original_min_image_size = self.min_image_size
                if ratio > 10:
                    self.min_image_size = int(original_min_image_size * 10 / ratio)
                    self.transforms = self.build_transform()

                with torch.cuda.device(self.dev):
                    inference_output = self.inference(image, obj)

                bboxes = inference_output.bbox.cpu().numpy().astype(int)
                # bboxes = self.to_left_right_upper_lower(bboxes)

                if ratio > 10:
                    self.min_image_size = original_min_image_size
                    self.transforms = self.build_transform()

                bboxes = torch.tensor(bboxes)

                # Convert to [left, lower, right, upper] instead of [left, upper, right, lower]
                height = image.shape[-2]
                bboxes = torch.stack([bboxes[:, 0], height - bboxes[:, 3], bboxes[:, 2], height - bboxes[:, 1]], dim=1)

                if confidence_threshold is not None:
                    self.confidence_threshold = original_confidence_threshold
                if return_labels:
                    # subtract 1 because it's 1-indexed for some reason
                    return bboxes, inference_output.get_field("labels").cpu().numpy() - 1
                return bboxes

        self.glip_demo = CustomGLIPDemo(*args, dev=self.dev, hp = self.hp)

    def forward(self, *args, **kwargs):
        return self.glip_demo.forward(*args, **kwargs)
"""

    
class XVLM(BaseModel):
    name = 'xvlm'
    load_order=0

    def __init__(self, gpu_number=0):

        from viper.src.base_models.xvlm.xvlm import XVLMBase
        from transformers import BertTokenizer

        super().__init__(gpu_number)

        image_res = 384
        self.max_words = 30
        config_xvlm = {
            'image_res': image_res,
            'patch_size': 32,
            'text_encoder': config.text_encoder.model,
            'block_num': 9,
            'max_tokens': 40,
            'embed_dim': 256,
        }

        vision_config = {
            'vision_width': 1024,
            'image_res': 384,
            'window_size': 12,
            'embed_dim': 128,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32]
        }
        
        
        if not config.use_local_models:
            warnings.warn("XVLM is not available through HuggingFace API. Attempting to load from local path.")
        
        with warnings.catch_warnings(), self.hp("XVLM"):
            model = XVLMBase(config_xvlm, use_contrastive_loss=True, vision_config=vision_config)
            checkpoint = torch.load(config.object_recognition.path + "/retrieval_mscoco_checkpoint_9.pth", map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
            msg = model.load_state_dict(state_dict, strict=False)
        if len(msg.missing_keys) > 0:
            print('XVLM Missing keys: ', msg.missing_keys)

        model = model.to(self.dev)
        model.eval()

        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained(config.text_encoder.path if config.use_local_models else config.text_encoder.model)

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_res, image_res), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        with open(f'{config.useful_lists_path}/random_negatives.txt') as f:
            self.negative_categories = [x.strip() for x in f.read().split()]

    @staticmethod
    def pre_caption(caption, max_words):
        caption = re.sub(
            r"([,.'!?\"()*#:;~])",
            '',
            caption.lower(),
        ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        if not len(caption):
            raise ValueError("pre_caption yields invalid text")

        return caption

    @torch.no_grad()
    def score(self, images, texts):

        if isinstance(texts, str):
            texts = [texts]

        if not isinstance(images, list):
            images = [images]

        images = [self.transform(image) for image in images]
        images = torch.stack(images, dim=0).to(self.dev)

        texts = [self.pre_caption(text, self.max_words) for text in texts]
        text_input = self.tokenizer(texts, padding='longest', return_tensors="pt").to(self.dev)

        image_embeds, image_atts = self.model.get_vision_embeds(images)
        text_ids, text_atts = text_input.input_ids, text_input.attention_mask
        text_embeds = self.model.get_text_embeds(text_ids, text_atts)

        image_feat, text_feat = self.model.get_features(image_embeds, text_embeds)
        logits = image_feat @ text_feat.t()

        return logits

    @torch.no_grad()
    def binary_score(self, image, text, negative_categories):
        # Compare with a pre-defined set of negatives
        texts = [text] + negative_categories
        sim = 100 * self.score(image, texts)[0]
        res = F.softmax(torch.cat((sim[0].broadcast_to(1, sim.shape[0] - 1),
                                   sim[1:].unsqueeze(0)), dim=0), dim=0)[0].mean()
        return res

    def forward(self, image, text, task='score', negative_categories=None):
        if task == 'score':
            score = self.score(image, text)
        else:  # binary
            score = self.binary_score(image, text, negative_categories=negative_categories)
        return score.cpu()
    
    
class GroundingDINO(BaseModel):
    name = 'dino'

    def __init__(self, gpu_number=0):
        super().__init__(gpu_number)

        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        
        model = config.object_detection.path if config.use_local_models else config.object_detection.model

        with warnings.catch_warnings(),self.hp("Grounding DINO"):
            
            max_memory = {gpu_number: torch.cuda.mem_get_info(self.dev)[0] if self.dev == f'cuda:{gpu_number}' else None}
            
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model,
                torch_dtype=torch.float16 if self.dev in {f'cuda:{gpu_number}',f'mps:{gpu_number}'} else torch.float32,
                low_cpu_mem_usage=True,
                max_memory=max_memory if self.dev == f'cuda:{gpu_number}' else None,
            ).to(self.dev)
            self.processor = AutoProcessor.from_pretrained(model, do_rescale=False)



    @torch.no_grad()
    def compute_prediction(self):
        pass

    @torch.no_grad()
    def forward(self, image, obj_names:list[str]):
        
        text = [part.strip() for name in obj_names for part in name.split('.') if part.strip()]  # Split by dot and remove empty parts
        
        text = [t for t in text if len(t) > 0]  # Remove empty strings
        
        final_text = ""
        
        for t in text:
            final_text += f"{t} . " 
        
        inputs = self.processor(images=image, text=final_text, return_tensors="pt").to(self.dev)
        
        outputs = self.model(**inputs)
        
        res = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=config.object_detection.detection_threshold,
            target_sizes=[(image.shape[1], image.shape[2])],
        )[0] # Retrieve first image results (only single image is processed at a time)
        
        bboxes = [[round(x,2) for x in box.tolist()] for box in res['boxes'].cpu().numpy()]
        labels = res['text_labels']
        
        return bboxes, labels
    
    
    
    
    
    
    
class SegmentAnything(BaseModel):
    name = 'sam'

    def __init__(self, gpu_number=0):
        super().__init__(gpu_number)

        from transformers import SamModel, SamProcessor
        
        with warnings.catch_warnings(), self.hp("Segment Anything"):
            
            self.model = SamModel.from_pretrained(
                config.object_segmentation.path if config.use_local_models else config.object_segmentation.model,
                torch_dtype=torch.float16 if self.dev in {f'cuda:{gpu_number}',f'mps:{gpu_number}'} else torch.float32,
                low_cpu_mem_usage=True,
            ).to(self.dev)
            
            self.processor = SamProcessor.from_pretrained(
                config.object_segmentation.path if config.use_local_models else config.object_segmentation.model,
                do_rescale=False
            )
            
                 
    @torch.no_grad()
    def forward(self, image, input_boxes:List[Tuple[int,int,int,int]]=None):
        """
        Forward pass for the Segment Anything model.
        :param image: The input image tensor.
        :param input_boxes: Optional bounding boxes for segmentation.
        :return: Segmentation masks and corresponding labels.
        """
        
        
        input_boxes = [[[float(dim) for dim in box] for box in input_boxes]] if input_boxes else None
        # List of List of Boxes, where each box is a list of coordinates [left, upper, right, lower]
        
        inputs = self.processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(torch.float32).to(self.dev)
        
        outputs = self.model(**inputs)
        
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
        scores = outputs.iou_scores.cpu()
        
        scores = scores.argmax(dim=2)[0]  
        
        return [mask[scores[idx].item()].unsqueeze(0) for idx, mask in enumerate(masks)]
    
    
    
    
    
    
    
    
    
cache = Memory(config.cache_path if config.use_cache else None, verbose=0)
@cache.cache(ignore=['result'])
def gpt_cache_aux(fn_name, image, question, temperature, result):
    """
    This is a trick to manually cache results from GPT. We want to do it manually because the queries to GPT are
    batched, and caching doesn't make sense for batches. With this we can separate individual samples in the batch
    """
    return result    
class GPT4VLM(OpenAIModel):
    name = 'gpt4vlm'

    def __init__(self, gpu_number=0):
        super().__init__(gpu_number)
        
        self.qa_prompt = "Question: {} Short Answer:"
        self.caption_prompt = "a photo of"
        self.max_tokens = 60
        self.temperature = config.object_captioning.temperature
        self.max_output_tokens = 60
        
    def pre_question(self, question):
        # from LAVIS blip_processors
        question = re.sub(
            r"([.!\"()*#:;~])",
            "",
            question.lower(),
        )
        question = question.rstrip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > self.max_tokens:
            question = " ".join(question_words[: self.max_tokens])

        return question
        
        
    
    def caption(self, image):
        
        responses = []
        
        for i in image:
        
            res = self.client.responses.create(
                model = config.object_captioning.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "input_text", "text": self.caption_prompt },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{i}",
                            },
                        ],
                    }
                ],
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ).output_text.strip()
        
            responses.append(res)
        
        return responses
    
    def qa(self, image, question):
        
        responses = []
        
        for i, q in zip(image,question):
        
            res = self.client.responses.create(
                model = config.object_captioning.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "input_text", "text": f"{q}" },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{i}",
                            },
                        ],
                    }
                ],
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ).output_text.strip()
            
            responses.append(res)
        
        return responses
        
        

    def forward(self, image, question=None, task='caption'):
        if not self.to_batch:
            image, question, task = [image], [question], [task]
            
        to_compute = None
        results = []
        # Check if in cache
        if config.use_cache:
            for idx,q in enumerate(question):
                # This is not ideal, because if not found, later it will have to re-hash the arguments.
                # But I could not find a better way to do it.
                result = gpt_cache_aux(task[idx], image[idx], q, self.temperature, None)
                results.append(result)  # If in cache, will be actual result, otherwise None
            to_compute = [i for i, r in enumerate(results) if r is None]
            question = [question[i] for i in to_compute]

        if len(question) > 0:
            # Separate into qa and caption batches.
            prompts_qa = [self.qa_prompt.format(self.pre_question(q)) for q, t in zip(question, task) if t == 'qa']
            
            images_qa = [im for i, im in enumerate(image) if task[i] == 'qa']
            images_caption = [im for i, im in enumerate(image) if task[i] == 'caption']

            response_qa = self.qa(images_qa, prompts_qa) if len(images_qa) > 0 else []
            response_caption = self.caption(images_caption) if len(images_caption) > 0 else []
                
            response = []
            for t in task:
                if t == 'qa':
                    response.append(response_qa.pop(0))
                else:
                    response.append(response_caption.pop(0))
        else:
            response = []  # All previously cached

        if config.use_cache:
            for idx, (q, r) in enumerate(zip(question, response)):
                # "call" forces the overwrite of the cache
                gpt_cache_aux.call(task[idx], image[idx],q, self.temperature, r)
            for i, idx in enumerate(to_compute):
                results[idx] = response[i]
        else:
            results = response

        if not self.to_batch:
            results = results[0]
        return results
        