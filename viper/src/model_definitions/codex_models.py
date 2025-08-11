
import warnings

from joblib import Memory

import torch
import backoff


from viper.src.utils import to_namespace

from .base_model import BaseModel
from .openai_model import OpenAIModel

from viper.configs import config


class CodexModel(BaseModel):
    name = 'codex'
    requires_gpu = False
    max_batch_size = 5
    load_order = 3

    # Not batched, but every call will probably be a batch (coming from the same process)

    def __init__(self, gpu_number=0):
        super().__init__(gpu_number=gpu_number)
        with open(config.codex.prompt) as f:
            self.base_prompt = f.read().strip()
            
        with open(config.codex.task_prompt) as f:
            self.task_prompt = f.read().strip()
        with open(config.codex.query_prompt) as f:
            self.query_prompt = f.read().strip()
        
        self.to_batch = False

    def forward(self, prompt, action, prompt_file=None, base_prompt=None, action_prompt=None, extra_context=None):

        if prompt_file is not None and base_prompt is None:  # base_prompt takes priority
            with open(prompt_file) as f:
                base_prompt = f.read().strip()
        elif base_prompt is None:
            base_prompt = self.base_prompt
        
        if action_prompt is None:
            if action == 'query':
                action_prompt = self.query_prompt
            elif action == 'task':
                action_prompt = self.task_prompt
            else:
                action_prompt = ""
            

        if isinstance(prompt, list):
            extended_prompt = [base_prompt.replace("INSERT_PROMPT_HERE", p).
                               replace('INSERT_ACTION_HERE', action_prompt).
                               replace('EXTRA_CONTEXT_HERE', str(ec))
                               for p, ec in zip(prompt, extra_context)]
        elif isinstance(prompt, str):
            extended_prompt = [base_prompt.replace("INSERT_PROMPT_HERE", prompt).
                               replace('INSERT_ACTION_HERE', action_prompt).
                               replace('EXTRA_CONTEXT_HERE', str(extra_context))]
        else:
            raise TypeError("prompt must be a string or a list of strings")

        result = self.forward_(extended_prompt)
        if not isinstance(prompt, list) and not self.to_batch:
            result = result[0]

        return result

    def forward_(self, extended_prompt):
        """
        This method should be implemented by subclasses to handle the actual model inference.
        It should take a list of extended prompts and return a list of responses.
        """
        raise NotImplementedError("Subclasses must implement this method.")


cache = Memory(config.cache_path if config.use_cache else None, verbose=0)
@cache.cache(ignore=['result'])
def o3_cache_aux(prompt, result):
    """
    This is a trick to manually cache results from GPT. We want to do it manually because the queries to GPT are
    batched, and caching doesn't make sense for batches. With this we can separate individual samples in the batch
    """
    return result
class O3Codex(CodexModel, OpenAIModel):
    name = 'o3codex'
    requires_gpu = False
    max_batch_size = 5
    load_order = 4
    
    def __init__(self, gpu_number=0):
        CodexModel.__init__(self, gpu_number=gpu_number)
        OpenAIModel.__init__(self, gpu_number=gpu_number)
        
        self.model = config.codex.model
        self.max_tokens = 5000
        self.reasoning_effort = 'low'  # Options: low, medium, high

        
        
    def run_codex(self, extended_prompt, max_tokens=2048):
        
        responses = []
        
        for p in extended_prompt:
                response = self.client.responses.create(
                    model=self.model,
                    input=[{"role": "user", "content": [{"type": "input_text", "text": p}]}],
                    max_output_tokens=max_tokens,
                    reasoning={"effort":self.reasoning_effort}
                )
                res = to_namespace(response)
                responses.append(res.output[-1].content[0].text)
        return responses
    
    
    def auto_batch(self, extended_prompt):
        """
        Automatically batches the extended prompts based on the max batch size.
        """
        if len(extended_prompt) > self.max_batch_size:
            response = []
            for i in range(0, len(extended_prompt), self.max_batch_size):
                response += self.auto_batch(extended_prompt[i:i + self.max_batch_size])
            return response
        else:
            return self.run_codex(extended_prompt, self.max_tokens)
            
        
    def forward_(self, extended_prompt):
        
        to_compute = None
        results = []
        # Check cache
        if config.use_cache:
            for prompt in extended_prompt:
                result = o3_cache_aux(prompt, result=None)
                results.append(result)
            to_compute = [i for i, r in enumerate(results) if r is None]
            extended_prompt = [extended_prompt[i] for i in to_compute]
            
        if len(extended_prompt) > 0:
            response = self.auto_batch(extended_prompt)
        else:
            response = []
            
        if config.use_cache:
            for p, r in zip(extended_prompt, response):
                o3_cache_aux.call(p, r)
            for i, idx in enumerate(to_compute):
                results[idx] = response[i]
        else:
            results = response
            
        return results   




cache = Memory(config.cache_path if config.use_cache else None, verbose=0)
@cache.cache(ignore=['result'])
def gpt_cache_aux(prompt, temperature, result):
    """
    This is a trick to manually cache results from GPT. We want to do it manually because the queries to GPT are
    batched, and caching doesn't make sense for batches. With this we can separate individual samples in the batch
    """
    return result   
class GPT4Codex(CodexModel, OpenAIModel):
    name = 'gpt4codex'
    requires_gpu = False
    max_batch_size = 5
    load_order = 4
    
    def __init__(self, gpu_number=0):
        CodexModel.__init__(self, gpu_number=gpu_number)
        OpenAIModel.__init__(self, gpu_number=gpu_number)
        
        self.temperature = config.codex.temperature
        self.model = config.codex.model
        self.api = config.codex.api
        self.max_tokens = 1024

        
        
    def run_codex(self, extended_prompt, max_tokens=256):
        
        responses = []
        
        for p in extended_prompt:
            if self.api == 'completions':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": p}],
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                )
                res = to_namespace(response)
                responses.append(res.choices[0].message.content)
            else:
                response = self.client.responses.create(
                    model=self.model,
                    input=[{"role": "user", "content": [{"type": "input_text", "text": p}]}],
                    max_output_tokens=max_tokens,
                    temperature=self.temperature,
                )
                res = to_namespace(response)
                responses.append(res.output[0].content[0].text)
        return responses
    
    
    def auto_batch(self, extended_prompt):
        """
        Automatically batches the extended prompts based on the max batch size.
        """
        if len(extended_prompt) > self.max_batch_size:
            response = []
            for i in range(0, len(extended_prompt), self.max_batch_size):
                response += self.auto_batch(extended_prompt[i:i + self.max_batch_size])
            return response
        else:
            return self.run_codex(extended_prompt,self.max_tokens)
            
        
    def forward_(self, extended_prompt):
        
        to_compute = None
        results = []
        # Check cache
        if config.use_cache:
            for prompt in extended_prompt:
                result = gpt_cache_aux(prompt, self.temperature, result=None)
                results.append(result)
            to_compute = [i for i, r in enumerate(results) if r is None]
            extended_prompt = [extended_prompt[i] for i in to_compute]
            
        if len(extended_prompt) > 0:
            response = self.auto_batch(extended_prompt)
        else:
            response = []
            
        if config.use_cache:
            for p, r in zip(extended_prompt, response):
                gpt_cache_aux.call(p, self.temperature, r)
            for i, idx in enumerate(to_compute):
                results[idx] = response[i]
        else:
            results = response
            
        return results
        
            



class CodeLlama(CodexModel):
    name = 'codellama'
    requires_gpu = True
    max_batch_size = 3
    load_order = 3  # Load this model last

    # Not batched, but every call will probably be a batch (coming from the same process)

    def __init__(self, gpu_number=0):
        super().__init__(gpu_number=gpu_number)

        from transformers import LlamaForCausalLM, CodeLlamaTokenizer

        # Load Llama2
        model_id = config.codex.path if config.use_local_models else config.codex.model

        self.tokenizer = CodeLlamaTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        # Compute this when the other models have already been loaded
        # Ignore gpu number
        usage_ratio = 0.15  # If it is small, it will use more GPUs, which will allow larger batch sizes
        leave_empty = 0.7  # If other models are using more than (1-leave_empty) of memory, do not use
        max_memory = {}
        for gpu_number in range(torch.cuda.device_count()):
            mem_available = torch.cuda.mem_get_info(f'cuda:{gpu_number}')[0]
            if mem_available <= leave_empty * torch.cuda.get_device_properties(gpu_number).total_memory:
                mem_available = 0
            max_memory[gpu_number] = mem_available * usage_ratio
            if gpu_number == 0:
                max_memory[gpu_number] /= 10
        self.model = LlamaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            # load_in_8bit=True,  # For some reason this results in OOM when doing forward pass
            device_map="sequential",
            max_memory=max_memory,
        )
        self.model.eval()

    def run_codellama(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        generated_ids = self.model.generate(input_ids.to("cuda"), max_new_tokens=128)
        generated_ids = generated_ids[:, input_ids.shape[-1]:]
        generated_text = [self.tokenizer.decode(gen_id, skip_special_tokens=True) for gen_id in generated_ids]
        generated_text = [text.split('\n\n')[0] for text in generated_text]
        return generated_text

    def forward_(self, extended_prompt):
        if len(extended_prompt) > self.max_batch_size:
            response = []
            for i in range(0, len(extended_prompt), self.max_batch_size):
                response += self.forward_(extended_prompt[i:i + self.max_batch_size])
            return response
        with torch.no_grad():
            response = self.run_codellama(extended_prompt)
        # Clear GPU memory
        torch.cuda.empty_cache()
        return response