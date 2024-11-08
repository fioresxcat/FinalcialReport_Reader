import os
import pdb

from modules.base_vllm import BaseModuleVLLM


class LLMExtractInfo(BaseModuleVLLM):
    def __init__(self, common_config, model_config):
        super().__init__(common_config, model_config)
    
        self._system_prompt = None
        self._fewshot_prompt = None
        
    
    def set_system_prompt(self, prompt):
        self._system_prompt = prompt

    
    @property
    def system_prompt(self):
        return self._system_prompt
    

    def set_fewshot_prompt(self, prompt):
        self._fewshot_prompt = prompt


    @property
    def fewshot_prompt(self):
        return self._fewshot_prompt
    

    def predict(self, inp_text):
        request = []
        if self._system_prompt is not None:
            request.append(self._system_prompt)
        if self._fewshot_prompt is not None:
            request.append(self._fewshot_prompt)
        request.append({"role": "user", "content": inp_text})

        chat_response = self.client.chat.completions.create(
            max_tokens=self.model_config['max_tokens'], 
            temperature=self.model_config['temperature'], 
            top_p=self.model_config['top_p'],
            model=self.model_config['lora_name'],
            messages = request,
        )
        return chat_response.choices[0].message.content.strip()
