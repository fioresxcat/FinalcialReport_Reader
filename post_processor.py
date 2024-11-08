from modules.base_vllm import BaseModuleVLLM
from utils import *
import os
import yaml
import unidecode
import pdb


class BCTCPostProcessor(BaseModuleVLLM):
    def __init__(self, common_config, model_config):
        super().__init__(common_config, model_config)
        my_dir = os.path.dirname(__file__)
        with open(os.path.join(my_dir, 'title_mapping.yaml')) as f:
            self.title_mapping = yaml.safe_load(f)


    def predict(self, request_id, inp, out, metadata):
        inp_data = inp.get_data()

        result = []

        # 1. add general info
        ocr_cands = {}
        for field_name, field_value in inp_data['general_infos'].items():
            if field_name not in ocr_cands:
                ocr_cands[field_name] = []
            ocr_cands[field_name].append(field_value)

        result.append({
            'group_name': 'general_info',
            'infos': ocr_cands
        })

        # 2. add doc info
        for doc_info in inp_data['list_docs']:
            ocr_cands = {}
            doc_general_infos = doc_info['extracted_infos']['general_infos']
            for field_name, field_value in doc_general_infos.items():
                if field_name not in ocr_cands:
                    ocr_cands[field_name] = []
                ocr_cands[field_name].append(field_value)

            doc_table_infos = doc_info['extracted_infos']['table_infos']
            ocr_cands['table'] = [doc_table_infos]

            result.append({
                'group_name': 'doc_info',
                'infos': ocr_cands
            })

        # pdb.set_trace()
        metadata = self.add_metadata(metadata, 1, 1)
        out.set_data(result)
        return out, metadata
    
        """
            result:
            {
                'general_infos': ,
                'doc_infos': [
                    {

                    }
                ]
            }
        """