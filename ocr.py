import os
import pdb
import cv2
import unidecode
import Levenshtein
import numpy as np
from PIL import Image

from modules.base import BaseModule
from modules.ocr_parseq.general_ocr import GeneralOCR


class BCTCOCR(GeneralOCR):
    instance = None
    
    def __init__(self, common_config, model_config):
        super(BCTCOCR, self).__init__(common_config, model_config)
        self.general_ocr = GeneralOCR.get_instance(common_config, model_config)
        
        
    @staticmethod
    def get_instance(common_config, model_config):
        if BCTCOCR.instance is None:
            BCTCOCR.instance = BCTCOCR(common_config, model_config)
        return BCTCOCR.instance
    
    
    def predict(self, request_id, inp, out, metadata):
        result = inp.get_data()

        # -------------- ocr for table text --------------
        indexes = []
        list_boxes = []
        for page_index, page_data in enumerate(result['pages']):
            if not page_data['has_table']:
                continue
            table_data = page_data['tables']
            table_data['raw_words'] = []
            table_data['raw_cands'] = []
            for table_index, list_rois in enumerate(table_data['text_images']):
                table_data['raw_words'].append([])
                table_data['raw_cands'].append([])
                for roi_index, roi in enumerate(list_rois):
                    indexes.append((page_index, table_index, roi_index))
                    list_boxes.append(roi)
                    table_data['raw_words'][table_index].append(None)
                    table_data['raw_cands'][table_index].append(None)

        if self.general_ocr.__class__.__name__ not in metadata:
            metadata[self.general_ocr.__class__.__name__] = {
                'num_request': 0,
                'total_batch_size': 0
            }

        list_raw_words, list_raw_cands = self.general_ocr.predict_batch([list_boxes], metadata)
        list_raw_words, list_raw_cands = list_raw_words[0], list_raw_cands[0]
        
        # add to raw_words and raw_cands
        for (page_index, table_index, roi_index), raw_word, raw_cand in zip(indexes, list_raw_words, list_raw_cands):
            table_data = result['pages'][page_index]['tables']
            table_data['raw_words'][table_index][roi_index] = raw_word
            table_data['raw_cands'][table_index][roi_index] = raw_cand
        result['charset_list'] = self.charset_list
        
        
        # change to Son format
        for page_index, page_data in enumerate(result['pages']):
            if not page_data['has_table']:
                continue
            table_data = page_data['tables']
            table_data['text'] = []
            for table_index, list_rois in enumerate(table_data['text_images']):
                table_data['text'].append([])
                for box, roi, text in zip(table_data['text_boxes'][table_index], list_rois, table_data['raw_words'][table_index]):
                    table_data['text'][table_index].append({'box':box, 'roi':roi, 'text':text})
        

        # -------------- ocr for outside table text --------------
        indexes = []
        list_boxes = []
        for page_index, page_data in enumerate(result['pages']):
            page_data['raw_words'] = []
            page_data['raw_cands'] = []
            if not page_data['has_table']:
                continue
            for roi_index, roi in enumerate(page_data['text_images']):
                indexes.append((page_index, roi_index))
                list_boxes.append(roi)
                page_data['raw_words'].append(None)
                page_data['raw_cands'].append(None)
        list_raw_words, list_raw_cands = self.general_ocr.predict_batch([list_boxes], metadata)
        list_raw_words, list_raw_cands = list_raw_words[0], list_raw_cands[0]
        for (page_index, roi_index), raw_word, raw_cand in zip(indexes, list_raw_words, list_raw_cands):
            page_data = result['pages'][page_index]
            page_data['raw_words'][roi_index] = raw_word
            page_data['raw_cands'][roi_index] = raw_cand

        
        # # debug plot
        # for page_index, page_data in enumerate(result['pages']):
        #     if not page_data['has_table']:
        #         continue
        #     print('RAW WORDS: ', page_data['raw_words'])
        #     pdb.set_trace()

        out.set_data(result)
        return out, metadata
