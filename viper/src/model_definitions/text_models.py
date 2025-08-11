import warnings
import os
from collections import Counter
from itertools import chain, repeat


from joblib import Memory




from .openai_model import OpenAIModel

from viper.src.utils import to_namespace
from viper.configs import config


cache = Memory(config.cache_path if config.use_cache else None, verbose=0)

@cache.cache(ignore=['result'])
def gpt_cache_aux(fn_name, prompts, temperature, n_votes, result):
    """
    This is a trick to manually cache results from GPT. We want to do it manually because the queries to GPT are
    batched, and caching doesn't make sense for batches. With this we can separate individual samples in the batch
    """
    return result
class GPT4(OpenAIModel):
    name = 'gpt4'
    to_batch = False
    requires_gpu = False
        

    def __init__(self, gpu_number=0):
        super().__init__(gpu_number=gpu_number)
        with open(config.llm.qa_prompt) as f:
            self.qa_prompt = f.read().strip()
        with open(config.llm.guess_prompt) as f:
            self.guess_prompt = f.read().strip()
        self.temperature = config.llm.temperature
        self.n_votes = config.llm.n_votes
        self.model = config.llm.model
        self.api = config.llm.api
            
    
            
        

    # initial cleaning for reference QA results
    @staticmethod
    def process_answer(answer):
        answer = answer.lstrip()  # remove leading spaces (our addition)
        answer = answer.replace('.', '').replace(',', '').lower()
        to_be_removed = {'a', 'an', 'the', 'to', ''}
        answer_list = answer.split(' ')
        answer_list = [item for item in answer_list if item not in to_be_removed]
        return ' '.join(answer_list)

    @staticmethod
    def get_union(lists):
        return list(set(chain.from_iterable(lists)))

    @staticmethod
    def most_frequent(answers):
        answer_counts = Counter(answers)
        return answer_counts.most_common(1)[0][0]

    def process_guesses(self, prompts):
        prompt_base = self.guess_prompt
        prompts_total = []
        
        n_prompts = list(chain.from_iterable(repeat(p, self.n_votes) for p in prompts))
        
        for p in n_prompts:
            question, guess1, _ = p
            if len(guess1) == 1:
                # In case only one option is given as a guess
                guess1 = [guess1[0], guess1[0]]
            prompts_total.append(prompt_base.format(question, guess1[0], guess1[1]))
        response = self.process_guesses_fn(prompts_total)
        
        if self.n_votes > 1:
            response_ = []
            for i in range(len(prompts)):
                if self.api == "completions":
                    resp_i = [r.message.content for r in
                              response.choices[i * self.n_votes:(i + 1) * self.n_votes]]
                else:
                    resp_i = [r.text for r in response.output[0].content[i * self.n_votes:(i + 1) * self.n_votes]]
                response_.append(self.most_frequent(resp_i).lstrip())
            response = response_
        else:
            if self.api == "completions":
                response = [r.message.content.lstrip() for r in response.choices]
            else:
                response = [r.text.lstrip() for r in response.output[0].content]
        return response

    def process_guesses_fn(self, prompt):
        # The code is the same as get_qa_fn, but we separate in case we want to modify it later
        response = self.query_model(prompt, model=self.model, max_tokens=16, logprobs=1, stream=False,
                                   stop=["\n", "<|endoftext|>"])
        return response

    def get_qa(self, prompts, prompt_base: str = None) -> list[str]:
        if prompt_base is None:
            prompt_base = self.qa_prompt
        prompts_total = []
        
        n_prompts = list(chain.from_iterable(repeat(p, self.n_votes) for p in prompts))
        
        for p in n_prompts:
            question = p
            prompts_total.append(prompt_base.format(question))
        response = self.get_qa_fn(prompts_total)
        if self.n_votes > 1:
            response_ = []
            for i in range(len(prompts)):
                if self.api == "completions":
                    resp_i = [r.message.content for r in
                              response.choices[i * self.n_votes:(i + 1) * self.n_votes]]
                else:
                    resp_i = [r.text for r in response.output[0].content[i * self.n_votes:(i + 1) * self.n_votes]]
                response_.append(self.most_frequent(resp_i))
            response = response_
        else:
            if self.api == "completions":
                response = [r.message.content for r in response.choices]
            else:
                response = [self.process_answer(r.text) for r in response.output[0].content]
        return response

    def get_qa_fn(self, prompt):
        response = self.query_model(prompt, model=self.model, max_tokens=16, logprobs=1, stream=False,
                                   stop=["\n", "<|endoftext|>"])
        return response

    def get_general(self, prompts) -> list[str]:
        response = self.query_model(prompts, model=self.model, max_tokens=256, top_p=1, frequency_penalty=0,
                                   presence_penalty=0)
        if self.api == "completions":
            response = [r.message.content for r in response.choices]
        else:
            response = [r.text for r in response.output[0].content]
        return response

    def query_model(self, prompt, model="gpt-4o-mini", max_tokens=16, logprobs=1, stream=False,
                   stop=None, top_p=1, frequency_penalty=0, presence_penalty=0):
        
        responses = {"choices": []} if self.api == "completions" else {"output": [{"content": []}]}
        
        for p in prompt:
        
            if self.api == "completions":
                messages = [{"role": "user", "content": p}]
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=self.temperature,
                )
                responses["choices"].append({"message":{"content":response.choices[0].message.content}})
            else:
                response = self.client.responses.create(
                    model=model,
                    input=[{"role": "user", "content": [{"type": "input_text", "text": p}]}],
                    max_output_tokens=max_tokens,
                    temperature=self.temperature,
                    stream=stream,
                    top_p=top_p,
                )
                responses["output"][0]["content"].append({"text":response.output[0].content[0].text})
        return to_namespace(responses)

    def forward(self, prompt, process_name):
        if not self.to_batch:
            prompt = [prompt]

        if process_name == 'gpt4_qa':
            # if items in prompt are tuples, then we assume it is a question and context
            if isinstance(prompt[0], tuple) or isinstance(prompt[0], list):
                prompt = [f'{question} {context}' for question, context in prompt]

        to_compute = None
        results = []
        # Check if in cache
        if config.use_cache:
            for p in prompt:
                # This is not ideal, because if not found, later it will have to re-hash the arguments.
                # But I could not find a better way to do it.
                result = gpt_cache_aux(process_name, p, self.temperature, self.n_votes, None)
                results.append(result)  # If in cache, will be actual result, otherwise None
            to_compute = [i for i, r in enumerate(results) if r is None]
            prompt = [prompt[i] for i in to_compute]

        if len(prompt) > 0:
            if process_name == 'gpt4_qa':
                response = self.get_qa(prompt)
            elif process_name == 'gpt4_guess':
                response = self.process_guesses(prompt)
            else:  # 'gpt3_general', general prompt, has to be given all of it
                response = self.get_general(prompt)
        else:
            response = []  # All previously cached

        if config.use_cache:
            for p, r in zip(prompt, response):
                # "call" forces the overwrite of the cache
                gpt_cache_aux.call(process_name, p, self.temperature, self.n_votes, r)
            for i, idx in enumerate(to_compute):
                results[idx] = response[i]
        else:
            results = response

        if not self.to_batch:
            results = results[0]
        return results

    @classmethod
    def list_processes(cls):
        return ['gpt4_qa', 'gpt4_guess', 'gpt4_general']
