import pdb
import os
from pathlib import Path
import yaml
import argparse
import fitz
import cv2
import numpy as np

from inpout import Input, Output
from processor import BCTCProcessor


def test_matching():
    import Levenshtein
    import unidecode
    import re
    from concurrent.futures import ThreadPoolExecutor
    import time

    def str_similarity(str1, str2):
        distance = Levenshtein.distance(str1, str2)
        score = 1 - (distance / (max(len(str1), len(str2))))
        return score


    paragraph = ( 
        'CTY CP ÁNH DƯƠNG VIỆT NAM Mẫu số B 01 - DN 648 Nguyễn Trãi, P.11, Q.5, TP.HCM BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT Tại ngày 30 tháng 6 năm 2015 Đơn vị tính: Đồng Việt Nam'
        'CTY CP ÁNH DƯƠNG VIỆT NAM Mẫu số B 01 - DN 648 Nguyễn Trãi, P.11, Q.5, TP.HCM BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT Tại ngày 30 tháng 6 năm 2015 Đơn vị tính: Đồng Việt Nam'
        'CTY CP ÁNH DƯƠNG VIỆT NAM Mẫu số B 01 - DN 648 Nguyễn Trãi, P.11, Q.5, TP.HCM BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT Tại ngày 30 tháng 6 năm 2015 Đơn vị tính: Đồng Việt Nam'
        'CTY CP ÁNH DƯƠNG VIỆT NAM Mẫu số B 01 - DN 648 Nguyễn Trãi, P.11, Q.5, TP.HCM BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT Tại ngày 30 tháng 6 năm 2015 Đơn vị tính: Đồng Việt Nam'
        'CTY CP ÁNH DƯƠNG VIỆT NAM Mẫu số B 01 - DN 648 Nguyễn Trãi, P.11, Q.5, TP.HCM BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT Tại ngày 30 tháng 6 năm 2015 Đơn vị tính: Đồng Việt Nam'
        'CTY CP ÁNH DƯƠNG VIỆT NAM Mẫu số B 01 - DN 648 Nguyễn Trãi, P.11, Q.5, TP.HCM BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT Tại ngày 30 tháng 6 năm 2015 Đơn vị tính: Đồng Việt Nam'
        'CTY CP ÁNH DƯƠNG VIỆT NAM Mẫu số B 01 - DN 648 Nguyễn Trãi, P.11, Q.5, TP.HCM BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT Tại ngày 30 tháng 6 năm 2015 Đơn vị tính: Đồng Việt Nam'
        'CTY CP ÁNH DƯƠNG VIỆT NAM Mẫu số B 01 - DN 648 Nguyễn Trãi, P.11, Q.5, TP.HCM BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT Tại ngày 30 tháng 6 năm 2015 Đơn vị tính: Đồng Việt Nam'
        'CTY CP ÁNH DƯƠNG VIỆT NAM Mẫu số B 01 - DN 648 Nguyễn Trãi, P.11, Q.5, TP.HCM BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT Tại ngày 30 tháng 6 năm 2015 Đơn vị tính: Đồng Việt Nam'
        'CTY CP ÁNH DƯƠNG VIỆT NAM Mẫu số B 01 - DN 648 Nguyễn Trãi, P.11, Q.5, TP.HCM BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT Tại ngày 30 tháng 6 năm 2015 Đơn vị tính: Đồng Việt Nam'
    )
    anchor = 'BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT'

    s = time.perf_counter()
    paragraph_words = paragraph.split()
    anchor_words = anchor.split()
    anchor_len = len(anchor_words)

    # Prepare list of n-grams to check
    n_grams = [" ".join(paragraph_words[i:i + anchor_len]) 
               for i in range(len(paragraph_words) - anchor_len + 1)]
    
    # Using ThreadPoolExecutor to parallelize the distance calculations
    similar_phrases = []
    num_threads = 1
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks and collect results
        future_to_ngram = {executor.submit(str_similarity, n_gram, anchor): n_gram for n_gram in n_grams}
        
    for future in future_to_ngram:
        if future.result() > 1:  # If similarity is above threshold
            similar_phrases.append(future_to_ngram[future])
            # break
    elapsed = time.perf_counter() - s
    print(similar_phrases)
    print('Elapsed time:', elapsed)
    pdb.set_trace()


def main(args):
    with open('configs/config_models.yaml') as f:
        config_models = yaml.safe_load(f)

    with open('configs/config_env.yaml') as f:
        config_env = yaml.safe_load(f)

    processor = BCTCProcessor(config_env, config_models)

    for fp in Path(args.inp_path).glob('*'):
        if 'ey' not in fp.stem:
            continue
        print(f'PROCESSING {fp} ...')
        if fp.suffix in ['.pdf', '.PDF']:
            images = []
            mat = fitz.Matrix(2, 2)
            docs = fitz.open(fp)
            for page in docs:
                pix = page.get_pixmap(matrix=mat)
                shape = (pix.height, pix.width, 3)
                image = np.ndarray(shape, dtype=np.uint8, buffer=pix.samples)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
        else:
            images = []
        
        images = images[:args.max_pages]
        inp = Input({'rotated_images': images})
        out, metadata = processor.predict(str(fp), inp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_path', default='test_files')
    parser.add_argument('--max_pages', type=int, default='100')
    args = parser.parse_args()
    main(args)

    # test_matching()