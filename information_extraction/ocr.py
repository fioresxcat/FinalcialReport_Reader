from modules.ocr_parseq.base_ocr import BaseOCR
import numpy as np


class OCR(BaseOCR):
    def predict_batch(self, list_list_boxes):
        batch_images = []
        page_lengths = []
        list_raw_words = []
        list_raw_cands = []
        for i in range(len(list_list_boxes)):
            list_raw_words.append([])
            list_raw_cands.append([])
            page_lengths.append(len(list_list_boxes[i]))
            for j, image in enumerate(list_list_boxes[i]):
                resized_image = self.resize(image)
                processed_image = np.transpose(resized_image/255., (2, 0, 1)).astype(np.float32)
                normalized_image = (processed_image - 0.5) / 0.5
                batch_images.append(normalized_image)

        batch_images_length = len(batch_images)
        batch_images = np.array(batch_images)
        text_output = []
        if len(batch_images) != 0:
            index = 0
            while index < len(batch_images):
                #print(len(batch_images[index:index+self.model_config['max_batch_size']]))
                text_output += self.request_batch(batch_images[index:index+self.model_config['max_batch_size']])
                index += self.model_config['max_batch_size']
        text_output = text_output[:batch_images_length]
        
        cnt_index = 0
        for i, page_length in enumerate(page_lengths):
            list_raw_cands[i] = text_output[cnt_index:cnt_index+page_length]
            for j in range(page_length):
                list_raw_words[i].append(self.index_to_word(text_output[cnt_index+j]))
            cnt_index += page_length
        return list_raw_words, list_raw_cands
    