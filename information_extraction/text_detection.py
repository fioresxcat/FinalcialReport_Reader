import pdb
import cv2
import numpy as np

from modules.base import BaseModule
from modules.text_detection_db.base_text_detection import BaseTextDetector
from utils import poly2box, iou_bbox, sort_polys


class BCTCTextDetector(BaseTextDetector):
    def __init__(self, common_config, model_config):
        super(BCTCTextDetector, self).__init__(common_config, model_config)


    def is_poly_belong(self, text_poly, block):
        text_bb = poly2box(text_poly)    
        r1, r2, iou = iou_bbox(text_bb, block)
        return r1 >= 0.5 


    def predict(self, page_img):
        h, w = page_img.shape[:2]
        image = self.resize_image(page_img, image_short_side=640)
        image = np.expand_dims(image[..., ::-1], 0)
        image = image.transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        image_input = np.ascontiguousarray(image, dtype='uint8')  # contiguous
        output_dict = self.request(image_input)
        p = np.array(output_dict.as_numpy(self.model_config['output_name']))[0]
        p = p.transpose((1, 2, 0))
        bitmap = p > 0.3
        bbs, scores = self.polygons_from_bitmap(p, bitmap, w, h, box_thresh=self.box_thresh, max_candidates=-1, unclip_ratio=1.2)
        
        bbs = np.reshape(bbs, (-1, 8)).tolist()
        sorted_bbs, _ = sort_polys(bbs)
        sorted_text_images = []
        for box_index, box in enumerate(sorted_bbs):
            box = np.array(box).reshape(4, 2)
            roi = self.four_point_transform(page_img, box)
            sorted_text_images.append(roi)

        # # debug plot
        # if len(bbs) > 0:
        #     tmp_img = page_img.copy()
        #     for bb in page_data['p4_bbs']:
        #         x1, y1, x2, y2 = list(map(int, bb))
        #         cv2.rectangle(tmp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.imwrite('test.jpg', tmp_img)
        #     pdb.set_trace()

        return sorted_bbs, sorted_text_images