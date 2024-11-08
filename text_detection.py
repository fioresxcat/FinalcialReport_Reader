import pdb
import cv2
import numpy as np

from modules.text_detection_db.base_text_detection import BaseTextDetector
from utils import poly2box


class BCTCTextDetector(BaseTextDetector):
    
    instance = None
    
    def __init__(self, common_config, model_config):
        super(BCTCTextDetector, self).__init__(common_config, model_config)

        
    @staticmethod
    def get_instance(common_config, model_config):
        if BCTCTextDetector.instance is None:
            BCTCTextDetector.instance = BCTCTextDetector(common_config, model_config)
        return BCTCTextDetector.instance


    def is_poly_belong(self, text, block):
        xmin_block, ymin_block, xmax_block, ymax_block = block
        if not isinstance(text, np.ndarray):
            text = np.array(text)
        xmin, xmax = np.min(text[:, 0]), np.max(text[:, 0])
        ymin, ymax = np.min(text[:, 1]), np.max(text[:, 1])
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        if x_center > xmin_block and x_center < xmax_block and y_center > ymin_block and y_center < ymax_block:
            return True
        return False


    def center(self, bb, axis):
        if axis == 'x':
            return (bb[0] + bb[2] + bb [4] + bb [6]) / 4
        elif axis == 'y':
            return (bb[1] + bb [3] + bb [5] + bb[7]) / 4


    def row_bbs(self, bbs):
        bbs.sort(key=lambda x: self.center(x, 'x'))
        clusters, mean_min, mean_max = [], [], []
        for bb in bbs:
            if len(clusters) == 0:
                clusters.append([bb])
                mean_min.append(bb[1])
                mean_max.append(bb[7])
                continue
            mid_y = self.center(bb, 'y')
            mid_x = self.center(bb, 'x')
            matched = None
            for idx, clt in enumerate(clusters):
                last_mid_x = self.center(clt[-1], 'x')
                if mean_min[idx] <= mid_y <= mean_max[idx]:
                    x_dist = mid_x - last_mid_x
                    if matched is None or x_dist < matched[1]:
                        matched = (idx, x_dist)
            if matched is None:
                clusters.append([bb])
                mean_min.append(bb[1])
                mean_max.append(bb[7])
            else:
                idx = matched[0]
                clusters[idx].append(bb)
                mean_min[idx] = (mean_min[idx] + bb[1]) / 2
                mean_max[idx] = (mean_max[idx] + bb[7]) / 2
        zip_clusters = list(zip(clusters, mean_min))
        zip_clusters.sort(key=lambda x: x[1])
        zip_clusters = list(np.array(zip_clusters, dtype=object)[:, 0])
    
        return zip_clusters


    def sort_bbs(self, bbs):
        bb_clusters = self.row_bbs(bbs)
        new_clusters = []
        for bb_cluster in bb_clusters:
            bb_cluster.sort(key=lambda b:b[0])
            new_clusters.append(bb_cluster)
        return new_clusters


    def predict(self, request_id, inp, out, metadata):
        result = inp.get_data()

        for page_index, (page_img, page_data) in enumerate(zip(result['refined_images'], result['pages'])):
            if not page_data['has_table']:
                continue
            
            # ---------------- infer ----------------
            h, w = page_img.shape[:2]
            image = self.resize_image(page_img, image_short_side=640)
            image = np.expand_dims(image[..., ::-1], 0)
            image = image.transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            image_input = np.ascontiguousarray(image, dtype='uint8')  # contiguous
            output_dict = self.request(image_input)
            metadata = self.add_metadata(metadata, 1, 1)
            p = np.array(output_dict.as_numpy(self.model_config['output_name']))[0]
            p = p.transpose((1, 2, 0))
            bitmap = p > 0.3
            bbs, scores = self.polygons_from_bitmap(p, bitmap, w, h, box_thresh=self.box_thresh, max_candidates=-1, unclip_ratio=1.2)
        
            # ---------------- assign to table ----------------
            table_data = page_data['tables']
            table_data['text_boxes'] = []
            table_data['text_images'] = []
            for i in range(len(table_data['images'])):
                table_data['text_boxes'].append([])
                table_data['text_images'].append([])

            if len(bbs) > 0:
                clusters = self.sort_bbs(np.reshape(bbs, (-1, 8)).tolist())
                text_boxes = np.concatenate(clusters, 0).reshape((-1, 4, 2))
                
                in_table_indexes = []
                for box_index, box in enumerate(text_boxes):
                    box = np.array(box)  # shape (4, 2)
                    for i, (table_box, table_image) in enumerate(zip(table_data['boxes'], table_data['images'])):
                        x1, y1, _, _ = table_box
                        if self.is_poly_belong(box, table_box):
                            box[..., 0] -= x1
                            box[..., 1] -= y1
                            table_data['text_boxes'][i].append(box)
                            roi = self.four_point_transform(table_image, box)
                            table_data['text_images'][i].append(roi)
                            in_table_indexes.append(box_index)
                            break
            
                # ---------------- text outside table ----------------
                ## get table ymin
                table_data = page_data['tables']
                min_table_ymin = min([bb[1] for bb in table_data['boxes']])
                page_data['p8_bbs'], page_data['text_images'], page_data['p4_bbs'] = [], [], []
                for box_index, box in enumerate(text_boxes):
                    p8_bb = box.flatten().tolist()
                    p4_bb = poly2box(box)
                    if box_index not in in_table_indexes and p4_bb[1] < min_table_ymin:
                        page_data['p8_bbs'].append(p8_bb)
                        roi = self.four_point_transform(page_img, box)
                        page_data['text_images'].append(roi)
                        page_data['p4_bbs'].append(p4_bb)
            else:
                page_data['p8_bbs'], page_data['text_images'], page_data['p4_bbs'] = [], [], []


            # # debug plot
            # if len(bbs) > 0:
            #     tmp_img = page_img.copy()
            #     for bb in page_data['p4_bbs']:
            #         x1, y1, x2, y2 = list(map(int, bb))
            #         cv2.rectangle(tmp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     cv2.imwrite('test.jpg', tmp_img)
            #     pdb.set_trace()

        out.set_data(result)
        return out, metadata

