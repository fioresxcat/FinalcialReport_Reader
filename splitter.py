from modules.base_vllm import BaseModuleVLLM
from utils import *
import os
import yaml
import unidecode
import pdb
import time


class BCTCSplitter(BaseModuleVLLM):
    def __init__(self, common_config, model_config):
        super().__init__(common_config, model_config)
        my_dir = os.path.dirname(__file__)
        with open(os.path.join(my_dir, 'title_mapping.yaml')) as f:
            self.title_mapping = yaml.safe_load(f)


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


    def new_doc(self):
        return {
            'images': [],
            'indexes': [],
            'doc_type': None,
            'raw_title': None,
        }

    def predict(self, request_id, inp, out, metadata):
        result = inp.get_data()

        list_docs = []
        doc_info = None
        last_doc_info = None
        founded_docs = []

        for page_index, (page_data, page_img) in enumerate(zip(result['pages'], result['refined_images'])):
            if not page_data['has_table']:
                continue
            page_text = ' '.join(page_data['raw_words'])
            # pdb.set_trace()
            # s = time.perf_counter()
            title_text, title_type = self.get_title_info(page_text)
            # print('TIME MATCH TITLE: ', time.perf_counter() - s)
            if title_type is None:  # trang ko co title
                if doc_info is not None:
                    if page_index - doc_info['indexes'][-1] == 1:
                        doc_info['page_indexes'].append(page_index)
                    else:
                        list_docs.append(doc_info)
                        doc_info = None
            
            elif title_type in ['BẢNG CÂN ĐỐI KẾ TOÁN', 'BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH', 'BÁO CÁO LƯU CHUYỂN TIỀN TỆ']:  # trang co title
                if doc_info is None:
                    doc_info = self.new_doc()
                    doc_info['images'].append(page_img)
                    doc_info['indexes'].append(page_index)
                    doc_info['doc_type'] = title_type
                    doc_info['raw_title'] = title_text
                else:
                    if title_type == doc_info['doc_type']:
                        doc_info['images'].append(page_img)
                        doc_info['indexes'].append(page_index)
                    else:
                        list_docs.append(doc_info)
                        doc_info = self.new_doc()
                        doc_info['images'].append(page_img)
                        doc_info['indexes'].append(page_index)
                        doc_info['doc_type'] = title_type
                        doc_info['raw_title'] = title_text

            elif title_type in ['BÁO CÁO CỦA BAN GIÁM ĐỐC', 'THUYẾT MINH BÁO CÁO TÀI CHÍNH']:
                pass


        if doc_info is not None:
            list_docs.append(doc_info)
        
        # ---------------- fill pages between doc -----------------
        list_docs.sort(key = lambda doc_info: doc_info['indexes'][0])
        for doc_index, doc_info in enumerate(list_docs):
            if doc_index == len(list_docs) - 1:
                break
            next_doc_info = list_docs[doc_index + 1]
            last_cur_index = doc_info['indexes'][-1]
            first_next_index = next_doc_info['indexes'][0]
            for page_index in range(last_cur_index + 1, first_next_index):
                doc_info['images'].append(result['refined_images'][page_index])
                doc_info['indexes'].append(page_index)
            doc_info['indexes'].sort()

        # ----------------- construct list docs ------------------
        for doc_info in list_docs:
            doc_info['tables'], doc_info['p4_bbs'], doc_info['p8_bbs'], doc_info['raw_words'], doc_info['raw_cands'] = [], [], [], [], []
            for page_index in doc_info['indexes']:
                page_data = result['pages'][page_index]
                doc_info['tables'].append(page_data['tables'])
                doc_info['p4_bbs'].append(page_data['p4_bbs'])
                doc_info['p8_bbs'].append(page_data['p8_bbs'])
                doc_info['raw_words'].append(page_data['raw_words'])
                doc_info['raw_cands'].append(page_data['raw_cands'])

        result['list_docs'] = list_docs

        # # debug
        # for doc_info in list_docs:
        #     print('\nDOCUMENT:')
        #     print('doc_type: ', doc_info['doc_type'])
        #     print('page_indexes: ', doc_info['indexes'])
        # pdb.set_trace()

        result.pop('pages')
        out.set_data(result)
        return out, metadata
    
        # result['pages']
        # page_data:
        # - tables: table infos
        # - has_table: bool
        # - p4_bbs:
        # - p8_bbs:  
        # - text_images: 
        # - raw_words:
        # - raw_cands: