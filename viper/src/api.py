from __future__ import annotations
import base64
import io

import numpy as np
import re
import torch
from dateutil import parser as dateparser
from PIL import Image
from rich.console import Console
from torchvision import transforms
from torchvision.ops import box_iou
from typing import Union, List
from word2number import w2n

from viper.configs import config
from viper.src.model_processes import forward
from viper.src.utils import load_json, show_single_image, zero_by_mask_and_aggregate

console = Console(highlight=False)


class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant
    information.
    Attributes
    ----------
    cropped_image : array_like
        An array-like of the cropped image taken from the original image.
    left : int
        An int describing the position of the left border of the crop's bounding box in the original image.
    lower : int
        An int describing the position of the bottom border of the crop's bounding box in the original image.
    right : int
        An int describing the position of the right border of the crop's bounding box in the original image.
    upper : int
        An int describing the position of the top border of the crop's bounding box in the original image.

    Methods
    -------
    find(object_name: str)->List[ImagePatch]
        Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the
        image matching the object_name.
    exists(object_name: str)->bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    verify_property(property: str)->bool
        Returns True if the property is met, and False otherwise.
    best_text_match(option_list: List[str], prefix: str)->str
        Returns the string that best matches the image.
    simple_query(question: str=None)->str
        Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?".
    compute_depth()->float
        Returns the median depth of the image crop.
    crop(left: int, lower: int, right: int, upper: int)->ImagePatch
        Returns a new ImagePatch object containing a crop of the image at the given coordinates.
    """

    def __init__(self, image: Union[Image.Image, torch.Tensor, np.ndarray], left: int = None, upper: int = None,
                 right: int = None, lower: int = None, parent_left=0, parent_upper=0, queues=None,
                 parent_img_patch=None,
                 alpha:Union[torch.Tensor, np.ndarray, None] = None):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        attributes. If no coordinates are provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.

        Parameters
        -------
        image : array_like
            An array-like of the original image.
        left : int
            An int describing the position of the left border of the crop's bounding box in the original image.
        lower : int
            An int describing the position of the bottom border of the crop's bounding box in the original image.
        right : int
            An int describing the position of the right border of the crop's bounding box in the original image.
        upper : int
            An int describing the position of the top border of the crop's bounding box in the original image.
        alpha : array_like, optional

        """

        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image).permute(1, 2, 0)
        elif isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            image = image / 255
            
        if isinstance(alpha, torch.Tensor) and image.dtype == torch.uint8:
            alpha = alpha/255
        elif isinstance(alpha, np.ndarray):
            alpha = torch.tensor(alpha).permute(1, 2, 0)
            
        if alpha is None and image.shape[0] == 4:
            alpha = image[3:4, :, :]
            image = image[:3, :, :]

        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.alpha = alpha
            self.left = 0
            self.upper = 0
            self.right = image.shape[2]  # width
            self.lower = image.shape[1]  # height
        else:
            self.cropped_image = image[:, upper:lower, left:right]
            self.alpha = alpha[:, upper:lower, left:right] if alpha is not None else None
            self.left = left + parent_left
            self.upper = upper + parent_upper
            self.right = right + parent_left
            self.lower = lower + parent_upper

        self.height = self.cropped_image.shape[1]
        self.width = self.cropped_image.shape[2]

        self.cache = {}
        self.queues = (None, None) if queues is None else queues

        self.parent_img_patch = parent_img_patch

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2
        
        self.bbox = (self.left, self.upper, self.right, self.lower)

        if self.cropped_image.shape[1] == 0 or self.cropped_image.shape[2] == 0:
            raise Exception("ImagePatch has no area")

        self.possible_options:dict = load_json(config.useful_lists_path+"/possible_options.json")

    def forward(self, model_name, *args, **kwargs):
        return forward(model_name, *args, queues=self.queues, **kwargs)


    def to_pil_image(self) -> Image.Image:
        """Converts the cropped image to a PIL Image."""
        if self.alpha is not None:
            img = torch.cat([self.cropped_image, self.alpha], dim=0)
            return transforms.ToPILImage()(img)
        
        return transforms.ToPILImage()(self.cropped_image)
    
    
    def apply_mask(self, alpha: torch.Tensor, replace: bool=False) -> ImagePatch:
        """Updates the alpha mask of the ImagePatch."""
        if alpha.shape[1:] != self.cropped_image.shape[1:]:
            raise ValueError(f"Alpha channel shape {alpha.shape[1:]} does not match image shape {self.cropped_image.shape[1:]}.")
        
        a = alpha if replace or self.alpha is None else torch.min(self.alpha, alpha)
        
        return ImagePatch(self.cropped_image, self.left, self.upper, self.right, self.lower, queues=self.queues,
                          parent_img_patch=self, alpha=a)
        



    def crop(self, left: int, upper: int, right: int, lower: int) -> ImagePatch:
        """Returns a new ImagePatch containing a crop of the original image at the given coordinates.
        Parameters
        ----------
        left : int
            the position of the left border of the crop's bounding box in the original image
        lower : int
            the position of the bottom border of the crop's bounding box in the original image
        right : int
            the position of the right border of the crop's bounding box in the original image
        upper : int
            the position of the top border of the crop's bounding box in the original image

        Returns
        -------
        ImagePatch
            a new ImagePatch containing a crop of the original image at the given coordinates
        """
        # make all inputs ints
        left = int(left)
        lower = int(lower)
        right = int(right)
        upper = int(upper)

        if config.crop_larger_margin:
            left = max(0, left - 10)
            upper = max(0, upper - 10)
            right = min(self.width, right + 10)
            lower = min(self.height, lower + 10)

        return ImagePatch(self.cropped_image, left, upper, right, lower, self.left, self.upper, queues=self.queues,
                          parent_img_patch=self, alpha=self.alpha)
            
        
        
    @property
    def original_image(self):
        if self.parent_img_patch is None:
            return self.cropped_image
        else:
            return self.parent_img_patch.original_image

    def find(self, object_name:str, mask:bool=False) -> list[ImagePatch]:
        """Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        List[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop
        """

        if isinstance(object_name, str):
            object_name = [object_name]
        
        all_object_coordinates, labels = self.forward(config.object_detection.name, self.cropped_image, object_name)
        
        if len(all_object_coordinates) == 0:
            return []
        
        masks = None
        
        if mask:
            masks = self.forward(config.object_segmentation.name, self.cropped_image, all_object_coordinates)
        
        out = []
        
        for idx, (left, upper, right, lower) in enumerate(all_object_coordinates):
            new_patch = self.apply_mask(masks[idx]) if masks is not None else self
            new_img = new_patch.crop(left, upper, right, lower)
            
            out.append(new_img)

        
        return out
    
    def simple_query(self, question: str) -> str:
        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?". The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.
        """
        
        pil_img = self.to_pil_image()
        
        buf = io.BytesIO()
        
        pil_img.save(buf, format='PNG')
        
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return self.forward(config.object_captioning.name, img_str, question, task='qa')

    def exists(self, object_name) -> bool:
        """Returns True if the object specified by object_name is found in the image, and False otherwise.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        """
        if object_name.isdigit() or object_name.lower().startswith("number"):
            object_name = object_name.lower().replace("number", "").strip()

            object_name = w2n.word_to_num(object_name)
            answer = self.simple_query("What number is written in the image (in digits)?")
            return w2n.word_to_num(answer) == object_name

        patches = self.find([object_name])

        def process_qa(response):
            response = re.sub(
                r"([.!\"()*#:;~?])",
                "",
                response.lower()
            )
        
            return response

        filtered_patches = []
        for patch in patches:
            if "yes" in process_qa(patch.simple_query(f"Is there a {object_name}?")):
                filtered_patches.append(patch)
        return len(filtered_patches) > 0

    def _score(self, category: str, negative_categories=None, model=config.object_recognition.name) -> float:
        """
        Returns a binary score for the similarity between the image and the category.
        The negative categories are used to compare to (score is relative to the scores of the negative categories).
        """
        task = 'binary_score' if negative_categories is not None else 'score'
        res = self.forward(model, self.cropped_image, category, task=task, negative_categories=negative_categories)
        res = res.item()
        
        return res

    def _detect(self, category: str, thresh, negative_categories=None, model=config.object_recognition.name) -> bool:
        return self._score(category, negative_categories, model) > thresh

    def verify_property(self, object_name: str, attribute: str) -> bool:
        """Returns True if the object possesses the property, and False otherwise.
        Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead
        checking whether the object possesses the property.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        attribute : str
            A string describing the property to be checked.
        """
        name = f"{attribute} {object_name}"
        model = config.object_recognition.name
        negative_categories = [f"{att} {object_name}" for att in self.possible_options['attributes']]
        
        return self._detect(
            name,
            thresh=config.object_recognition.detection_threshold,
            negative_categories=negative_categories,
            model=model
        )

    def best_text_match(self, option_list: list[str] = None, prefix: str = None) -> str:
        """Returns the string that best matches the image.
        Parameters
        -------
        option_list : str
            A list with the names of the different options
        prefix : str
            A string with the prefixes to append to the options
        """
        option_list_to_use = option_list
        if prefix is not None:
            option_list_to_use = [prefix + " " + option for option in option_list]

        model_name = config.object_recognition.name
        image = self.cropped_image
        text = option_list_to_use
        res = self.forward(model_name, image, text, task='score')
        res = res.argmax().item()
        selected = res


        return option_list[selected]

    def compute_depth(self):
        """Returns the median depth of the image crop. Performs best when passed a masked image crop.
        If the crop is not masked, it will return the median depth of the entire crop.
        Parameters
        ----------
        Returns
        -------
        float
            the median depth of the image crop
        """
        original_image = self.original_image
        depth_map = self.forward(config.depth_estimation.name, original_image)
        depth_map = depth_map[self.upper:self.lower,
                              self.left:self.right]
        
        depth_agg = depth_map.median()
        
        if self.alpha is not None:
            _, depth_agg = zero_by_mask_and_aggregate(self.alpha.squeeze(0), depth_map, torch.median)
            
        return float(depth_agg.item())
        


    def overlaps_with(self, left, upper, right, lower):
        """Returns True if a crop with the given coordinates overlaps with this one,
        else False.
        Parameters
        ----------
        left : int
            the left border of the crop to be checked
        lower : int
            the lower border of the crop to be checked
        right : int
            the right border of the crop to be checked
        upper : int
            the upper border of the crop to be checked

        Returns
        -------
        bool
            True if a crop with the given coordinates overlaps with this one, else False
        """
        return self.left <= right and self.right >= left and self.lower >= upper and self.upper <= lower

    def llm_query(self, question: str, long_answer: bool = True) -> str:
        return llm_query(question, None, long_answer)

    def print_image(self, size: tuple[int, int] = None):
        show_single_image(self.cropped_image, size)

    def __repr__(self):
        return "ImagePatch(left={}, upper={}, right={}, lower={})".format(self.left, self.upper, self.right, self.lower)


def best_image_match(list_patches: list[ImagePatch], content: List[str], return_index: bool = False) -> \
        Union[ImagePatch, None]:
    """Returns the patch most likely to contain the content.
    Parameters
    ----------
    list_patches : List[ImagePatch]
    content : List[str]
        the object of interest
    return_index : bool
        if True, returns the index of the patch most likely to contain the object

    Returns
    -------
    int
        Patch most likely to contain the object
    """
    
    if len(list_patches) == 0:
        return None
    
    model = config.object_recognition.name
    
    scores = []
    
    for cont in content:
        res = forward(model, [p.cropped_image for p in list_patches], cont, task='score')
        
        scores.append(res)
        
    scores = torch.stack(scores).mean(dim=0) # Average over all contents
    best_index = scores.argmax().item()
    
    if return_index:
        return best_index
    
    return list_patches[best_index]


def distance(patch_a: Union[ImagePatch, float], patch_b: Union[ImagePatch, float]) -> float:
    """
    Returns the distance between the edges of two ImagePatches, or between two floats.
    If the patches overlap, it returns a negative distance corresponding to the negative intersection over union.
    """

    if isinstance(patch_a, ImagePatch) and isinstance(patch_b, ImagePatch):
        a_min = np.array([patch_a.left, patch_a.upper])
        a_max = np.array([patch_a.right, patch_a.lower])
        b_min = np.array([patch_b.left, patch_b.upper])
        b_max = np.array([patch_b.right, patch_b.lower])

        u = np.maximum(0, a_min - b_max)
        v = np.maximum(0, b_min - a_max)

        dist = np.sqrt((u ** 2).sum() + (v ** 2).sum())

        if dist == 0:
            box_a = torch.tensor([patch_a.left, patch_a.upper, patch_a.right, patch_a.lower])[None]
            box_b = torch.tensor([patch_b.left, patch_b.upper, patch_b.right, patch_b.lower])[None]
            dist = - box_iou(box_a, box_b).item()

    else:
        dist = abs(patch_a - patch_b)

    return dist


def bool_to_yesno(bool_answer: bool) -> str:
    """Returns a yes/no answer to a question based on the boolean value of bool_answer.
    Parameters
    ----------
    bool_answer : bool
        a boolean value

    Returns
    -------
    str
        a yes/no answer to a question based on the boolean value of bool_answer
    """
    return "yes" if bool_answer else "no"


def llm_query(query, context=None, long_answer=True, queues=None):
    """Answers a text question using GPT-4. The input question is always a formatted string with a variable in it.

    Parameters
    ----------
    query: str
        the text question to ask. Must not contain any reference to 'the image' or 'the photo', etc.
    """
    if long_answer:
        return forward(model_name='gpt4_general', prompt=query, queues=queues)
    else:
        return forward(model_name='gpt4_qa', prompt=[query, context], queues=queues)


def process_guesses(prompt, guess1=None, guess2=None, queues=None):
    return forward(model_name='gpt4_guess', prompt=[prompt, guess1, guess2], queues=queues)


def coerce_to_numeric(string, no_string=False):
    """
    This function takes a string as input and returns a numeric value after removing any non-numeric characters.
    If the input string contains a range (e.g. "10-15", "25to26"), it returns the first value in the range.
    """
    if any(month in string.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july',
                                                 'august', 'september', 'october', 'november', 'december']):
        try:
            return dateparser.parse(string).timestamp().year
        except:  # Parse Error
            pass

    try:
        # If it is a word number (e.g. 'zero')
        numeric = w2n.word_to_num(string)
        return numeric
    except ValueError:
        pass

    string_to = re.sub(r"\s+to\s+", "-", string, flags=re.IGNORECASE)

    # Remove any non-numeric characters except the decimal point and the negative sign
    string_re = re.sub(r"[^0-9\.\-]", "", string_to)

    if string_re.startswith('-'):
        string_re = '&' + string_re[1:]

    # Check if the string includes a range
    if "-" in string_re:
        # Split the string into parts based on the dash character
        parts = string_re.split("-")
        return coerce_to_numeric(parts[0].replace('&', '-'))
    else:
        string_re = string_re.replace('&', '-')

    try:
        # Convert the string to a float or int depending on whether it has a decimal point
        if "." in string_re:
            numeric = float(string_re)
        else:
            numeric = int(string_re)
    except:
        if no_string:
            raise ValueError
        # No numeric values. Return input
        return string
    return numeric
