import pdb
import cv2
import numpy as np
import os
import json
import json_repair
import unidecode
from pathlib import Path
import yaml
import time

from modules.base import BaseModule
from llm_extract_info import LLMExtractInfo
from .prompt_llm.prompt import BCTCExtractionPrompt
from .table_structure import TableStructure
from .text_detection import BCTCTextDetector
from .ocr import OCR
from utils import poly2box, iou_bbox, sort_polys, str_similarity



class BCTCTableInformationExtraction:
    instance = None

    def __init__(self, common_config, model_config):
        # super(BCTCTableInformationExtraction, self).__init__(common_config, model_config)
        self.common_config = common_config
        self.model_config = model_config

        self.llm = LLMExtractInfo(common_config['vllm_server'], model_config['vllm_models']['base-model'])

        my_dir = Path(os.path.dirname(__file__))
        with open(os.path.join(my_dir, 'prompt_llm', 'bctc_input.json')) as f:
            json_input = json.load(f)
        prompt_template = BCTCExtractionPrompt(json_input)
        self.llm.set_system_prompt(prompt_template.get_prompt_system_vi())

        self.table_structure = TableStructure(common_config['triton_server'], model_config['triton_models']['table_structure'])
        self.text_detector = BCTCTextDetector(common_config['triton_server'], model_config['triton_models']['text_detection'])
        self.ocr = OCR(common_config['triton_server'], model_config['triton_models']['ocr'])

        with open(os.path.join(my_dir.parent, 'title_mapping.yaml')) as f:
            self.title_mapping = yaml.safe_load(f)

    @staticmethod
    def get_instance(common_config, model_config):
        if BCTCTableInformationExtraction.instance is None:
            BCTCTableInformationExtraction.instance = BCTCTableInformationExtraction(common_config, model_config)
        return BCTCTableInformationExtraction.instance
    

    
    def get_page_text(self, page_img):
        sorted_bbs, sorted_text_images = self.text_detector.predict(page_img)
        page_h, page_w = page_img.shape[:2]
        new_bbs, new_text_images = [], []
        for bb, text_image in zip(sorted_bbs, sorted_text_images):
            if min(bb[1::2]) <= page_h // 2:
                new_bbs.append(bb)
                new_text_images.append(text_image)
        list_raw_words, list_raw_cands = self.ocr.predict_batch([new_text_images])
        raw_words, raw_cands = list_raw_words[0], list_raw_cands[0]
        page_text = ' '.join(raw_words)
        return page_text


    def get_title_info(self, page_text, threshold=0.8, normalize=True, remove_space=False):
        if normalize:
            page_text = unidecode.unidecode(page_text)
        if remove_space:
            page_text = page_text.replace(' ', '')
        page_words = page_text.split()
        
        for title_type, title_list in self.title_mapping.items():
            for title_text in title_list:
                if normalize:
                    title_text = unidecode.unidecode(title_text)
                if remove_space:
                    title_text = title_text.replace(' ', '')
                # if title_text in page_text:
                #     return title_text, title_type

                title_words = title_text.split()
                title_len = len(title_words)
                for i in range(len(page_words) - title_len + 1):
                    n_gram = " ".join(page_words[i:i + title_len])
                    if str_similarity(n_gram, title_text) > threshold:
                        return title_text, title_type
                    
        return None, None


    def predict(self, request_id, inp, out, metadata):
        result = inp.get_data()
        list_docs = result['list_docs']

        # --------------------- init empty dât ---------------------
        for doc_info in list_docs:
            doc_info['extracted_infos'] = {
                'general_infos': None,
                'table_infos': None
            }
        result['general_infos'] = {
            'company_name': None,
            'address': None,
            'phone_number': None,
            'fax_number': None,
            'email': None,
            'currency': None,
        }

        # 1. --------------------- extract outside info using LLM ---------------------
        all_texts = []
        for doc_info in list_docs:
            first_page_words = doc_info['raw_words'][0]  # only take information for 1st page
            all_texts.append(' '.join(first_page_words))

        ## 1.2 get thuyetminh page text 
        thuyetminh_page_index = list_docs[-1]['indexes'][-1] + 1
        page_img = result['refined_images'][thuyetminh_page_index]
        page_text = self.get_page_text(page_img)
        # s = time.perf_counter()
        title_text, title_type = self.get_title_info(page_text)
        # print('TIME MATCH TITLE: ', time.perf_counter() - s)
        if title_type == 'THUYẾT MINH BÁO CÁO TÀI CHÍNH':
            all_texts.append(page_text)

        ## 1.3 get baocao text
        first_page_index = list_docs[0]['indexes'][0]
        for page_index in range(first_page_index - 1, -1, -1):
            page_img = result['refined_images'][page_index]
            page_text = self.get_page_text(page_img)
            # s = time.perf_counter()
            title_text, title_type = self.get_title_info(page_text)
            # print('TIME MATCH TITLE: ', time.perf_counter() - s)
            if title_type == 'BÁO CÁO CỦA BAN GIÁM ĐỐC':
                all_texts.append(page_text)
                break
        
        ## 1.3 concat all and request to LLM
        all_text = '\n\n'.join(all_texts)
        response = self.llm.predict(all_text)
        response = json_repair.loads(response)

        # update at whole pdf level
        for field_name, field_value in response.items():
            if field_name in result['general_infos'] and len(field_value) != '':
                result['general_infos'][field_name] = field_value
            elif 'company_name' in field_name and result['general_infos']['company_name'] is None:
                result['general_infos']['company_name'] = field_value

        for doc_info in list_docs:
            doc_type = doc_info['doc_type']
            if doc_type == 'BẢNG CÂN ĐỐI KẾ TOÁN':
                general_infos = {k: v for k, v in response.items() if k.startswith('CDKT_')}
            elif doc_type == 'BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH':
                general_infos = {k: v for k, v in response.items() if k.startswith('KQKD_')}
            elif doc_type == 'BÁO CÁO LƯU CHUYỂN TIỀN TỆ':
                general_infos = {k: v for k, v in response.items() if k.startswith('BCLCTT_')}
            doc_info['extracted_infos']['general_infos'] = general_infos
        

        # 2. --------------------- extract table info ---------------------
        for doc_info in list_docs:
            doc_info = self.table_structure.predict(doc_info)

        # pdb.set_trace()
        result['list_docs'] = list_docs
        out.set_data(result)
        return out, metadata
    

if __name__ == '__main__':
    import yaml

    with open('configs/config_models.yaml') as f:
        config_models = yaml.safe_load(f)

    with open('configs/config_env.yaml') as f:
        config_env = yaml.safe_load(f)

    module = BCTCTableInformationExtraction(config_env, config_models)
    pdb.set_trace()