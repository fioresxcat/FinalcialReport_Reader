import json
from pathlib import Path
import os

class BCTCExtractionPrompt:
    def __init__(self, json_input):
        self.doc_name = json_input["doc_name"]
        self.requirements = json_input["requirements"]
        self.field = list(self.requirements.keys())
        self.samples = json_input.get("samples", False)
        if not self.samples or len(self.samples) == 0:
            self.samples = []
        
    def get_prompt_assistant_vi(self):
        messages = []
        for sample in self.samples:
            assistant = {}
            content = {}
            
            data_txt = self.samples[sample]["raw_text"]
            js_content = self.samples[sample]["result"]
            for fd in self.field:
                content[fd] = js_content[fd]
            
            content= json.dumps(content, ensure_ascii=False)
                
            assistant = {}
            assistant['role'] = 'user'
            assistant['content'] = "Cho đoạn văn bản sau: " + "\"" +  data_txt + "\"" + "\n "
            messages.append(assistant)
            
            assistant = {}
            assistant['role'] = "assistant"
            assistant['content'] = "Trích xuất thông tin cần thiết: " + content + "\n "
            messages.append(assistant)
        
        return messages
        
    def get_prompt_assistant_en(self):
        messages = []
        for sample in self.samples:
            assistant = {}
            content = {}
            
            data_txt = self.samples[sample]["raw_text"]
            js_content = self.samples[sample]["result"]
            for fd in self.field:
                content[fd] = js_content[fd]
            
            content= json.dumps(content, ensure_ascii=False)
                
            assistant = {}
            assistant['role'] = 'user'
            assistant['content'] = "Given this text: " + "\"" +  data_txt + "\"" + "\n "
            messages.append(assistant)
            
            assistant = {}
            assistant['role'] = "assistant"
            assistant['content'] = "Extract necessary information: " + content + "\n "
            messages.append(assistant)
        
        return messages
    
    def get_prompt_system_vi(self):
        description_list = []
        output_type_list = []
        for field in self.requirements:
            
            description = self.requirements[field].get('description', '')
            if description != '':
                description_list.append(f"- {field}: {description}.")
            else:
                description_list.append(f"- {field}: {field}")
                
            if self.requirements[field]["output_type"] == 'string':
                output_type_list.append(f"\"{field}\": \"Kết quả\",")
            elif self.requirements[field]["output_type"] == 'list':
                output_type_list.append(f"\"{field}\": [\"Kết quả\"],")
                
        self.description = '\n        '.join(description_list)
        self.output_type = '\n        '.join(output_type_list)
        
        context = f"""
            Bạn là một trợ lý chỉ trả lời bằng tiếng Việt. Nhiệm vụ của bạn là trích xuất các thông tin trong {self.doc_name} bằng tiếng việt.
            Các thông tin cần trích xuất trong tài liệu bao gồm: '{"', '".join(self.field)}'.
        """        
        instruction = f"""Thông tin các trường cần trích xuất được định nghĩa như sau:
        {self.description}
        """
        output = f"""Hãy trả về thông tin trích xuất theo cấu trúc json sau:
        {{
        {self.output_type}
        }}.
        Tất cả cả trường đều phải được trả về. Nếu trường thông tin nào không có giá trị, trả về "" hoặc [] tùy theo định dạng của mỗi trường.
        Nếu trường thông tin nào bị sai về mặt chính tả hoặc ngữ nghĩa, sửa lại theo đúng ngôn ngữ Tiếng Việt.
        Không trả về các trường không được yêu cầu.
        """
        
        system_prompt = {
            'role': "system",
            'content': f"{context}\n\n{instruction}\n\n{output}"
        }
        
        return system_prompt
    
    def get_prompt_system_en(self):
        description_list = []
        output_type_list = []
        for field in self.requirements:
            
            description = self.requirements[field].get('description', '')
            if description != '':
                description_list.append(f"- {field}: {description}.")
            else:
                description_list.append(f"- {field}: {field}")
                
            if self.requirements[field]["output_type"] == 'string':
                output_type_list.append(f"\"{field}\": \"Result\"")
            elif self.requirements[field]["output_type"] == 'list':
                output_type_list.append(f"\"{field}\": [\"Result\"]")
        self.description = ',\n        '.join(description_list)
        self.output_type = ',\n        '.join(output_type_list)
        
        context = f"""
            You are an expert in extracting information, who only responds in English. Your task is to extract the information in {self.doc_name} in English.
            The information to be extracted from the document includes: '{"', '".join(self.field)}'.
        """        
        instruction = f"""The information of the fields to be extracted is defined as follows: {self.description}
        """
        output = f"""Please return the extracted information in the following json structure:
        {{ {self.output_type} }}.
        All fields must be returned. If any field has no value, return "" or [] depending on the format of each field.
        If any information field is incorrect in spelling or semantics, correct it.
        Do not return fields that are not required.
        """
        
        system_prompt = {
            'role': "system",
            'content': f"{context}\n\n{instruction}\n\n{output}"
        }

        return system_prompt
    