import pdb
import cv2
import numpy as np

from modules.base import BaseModule
from utils import iou_bbox, poly2box, sort_polys





class TableStructure(BaseModule):
    def __init__(self, common_config, model_config):
        super(TableStructure, self).__init__(common_config, model_config)
        self.labels = ['table', 'table_column', 'table_row', 'table column header', 'table projected row header', 'table spanning cell', 'no object']
        self.no_object_index = len(self.labels)-1
        self.MEAN, self.STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    
    def is_poly_belong(self, text_poly, block):
        text_bb = poly2box(text_poly)    
        r1, r2, iou = iou_bbox(text_bb, block)
        return r1 >= 0.5 


    def preprocess(self, image):
        h, w = image.shape[:2]
        current_max_size = max(w, h)
        scale = 1000 / current_max_size
        image = cv2.resize(image, (int(round(scale*w)), int(round(scale*h))))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # comment this line if image is already rgb ???

        im = image / 255.0
        im = (im - self.MEAN) / self.STD
        im = np.expand_dims(im, 0).transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im, dtype='float32')  # contiguous

        return im
    

    def correct_boxes(self, boxes, scores, class_names, text_boxes):
        ## postprocess name
        corrected_boxes, corrected_scores, corrected_class_names = [], [], []
        for b, s, c in zip(boxes, scores, class_names):
            has_text = False
            for poly in text_boxes:
                if self.is_poly_belong(poly, b):
                    has_text = True
                    break
            if has_text:
                if c == 'table': 
                    continue
                elif 'column' in c:
                    corrected_class_names.append('col')
                elif 'row' in c:
                    corrected_class_names.append('row')
                else:
                    corrected_class_names.append('span')
                corrected_boxes.append(b)
                corrected_scores.append(s)
        
        return corrected_boxes, corrected_scores, corrected_class_names
    

    def remove_overlapping_boxes(self, bboxes, scores, class_names, threshold=0.4):
        # Sort boxes by descending order of scores
        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        sorted_bboxes = [bboxes[i] for i in sorted_indices]
        sorted_class_names = [class_names[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        # List to store non-overlapping boxes
        non_overlapping_boxes = []
        non_overlapping_scores = []
        non_overlapping_class_names = []
        
        # Iterate through sorted boxes
        for bbox, score, class_name in zip(sorted_bboxes, sorted_scores, sorted_class_names):
            # Check for overlap with previously selected non-overlapping boxes
            if all(iou_bbox(bbox, prev_bbox)[2] <= threshold for prev_bbox in non_overlapping_boxes):
                non_overlapping_boxes.append(bbox)
                non_overlapping_scores.append(score)
                non_overlapping_class_names.append(class_name)
        
        return non_overlapping_boxes, non_overlapping_scores, non_overlapping_class_names


    def get_span_of_cell(self, cell, spans):
        for span in spans:
            if self.is_poly_belong(cell, span):
                return span
        return None


    def extract_cells(self, rows, cols, spans):
        cells = []
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                xr1, yr1, xr2, yr2 = row
                xc1, yc1, xc2, yc2 = col
                # cells.append({'box':[xc1, yr1, xc2, yr2], 'relation':[i, i+1, j, j+1]})
                # Now relation of a cell correspond to row and col
                cells.append({'box':[xc1, yr1, xc2, yr2], 'relation':[i, i, j, j]})
            
        ## Replace span cell into cells
        '''
        Idea: Xét 1 cell
            - nếu cell này ko thuộc về span cell nào -> Lấy
            - nếu cell này thuộc về 1 span cell:
                + nếu chưa lấy span cell của cell này 
                    --> Lấy span cell, cho relative của span cell chính là cell đang xét
                + nếu đã lấy spann cell của cell này 
                    --> Tăng relative của span cell lên
        '''
    
        if len(spans) > 0: 
            new_cells = []
            flags = {str(span):False for span in spans}
            for i, cell in enumerate(cells):
                span_of_cell = self.get_span_of_cell(cell['box'], spans)
                if span_of_cell is None:
                    new_cells.append(cell)
                    continue
                if not flags[str(span_of_cell)]:
                    new_cells.append({'box':span_of_cell, 'relation':cell['relation']})
                    flags[str(span_of_cell)] = True
                else:
                    idx = [k for k, cell in enumerate(new_cells) if str(cell['box'])==str(span_of_cell)][0]
                    sr = min(new_cells[idx]['relation'][0], cell['relation'][0])
                    er = max(new_cells[idx]['relation'][1], cell['relation'][1])
                    sc = min(new_cells[idx]['relation'][2], cell['relation'][2])
                    ec = max(new_cells[idx]['relation'][3], cell['relation'][3])
                    new_cells[idx]['relation'] = [sr, er, sc, ec]
        else:
            new_cells = cells
        
        return new_cells


    def texts2cells(self, texts, cells, table_bbox, page_shape):
        '''
        texts: a list, format of each element is {'box': ..., 'score':..., 'roi':..., 'text':...}
        cells: a list, format of each element is {'box': ..., 'relative': ...}
        '''
        mask = [0 for i in range(len(texts))]
        xmin, ymin = table_bbox[0], table_bbox[1]
        page_h, page_w = page_shape
        for cell in cells:
            cell_texts = [] 
            for i, text in enumerate(texts):
                if mask[i] == 1: continue
                if self.is_poly_belong(text['box'], cell['box']):
                    cell_texts.append(text)
                    mask[i] = 1

            # sort text
            if len(cell_texts) > 0:
                bb2text = {}
                bbs = []
                for text in cell_texts:
                    bb = text['box'].flatten().tolist()
                    bb2text[tuple(bb)] = text['text']
                    bbs.append(bb)
                sorted_bbs, _ = sort_polys(bbs)
                sorted_texts = [bb2text[tuple(bb)] for bb in sorted_bbs]
                cell['text'] = ' '.join(sorted_texts)
            else:
                cell['text'] = ''
                
            # adjust cell bbox
            cell['box'][0] += xmin
            cell['box'][1] += ymin
            cell['box'][2] += xmin
            cell['box'][3] += ymin

            # normalize
            cell['box'][0] /= page_w
            cell['box'][1] /= page_h
            cell['box'][2] /= page_w
            cell['box'][3] /= page_h


        return cells



    def predict_row_col(self, image):
        h, w = image.shape[:2]
        processed_img = self.preprocess(image)
        output_dict = self.request(processed_img)
        detections = np.array(output_dict.as_numpy(self.model_config['output_name']))  # shape (bs, 125, 11)
        detections = np.squeeze(detections, axis=0)
        boxes, scores, class_names = [], [], []
        for detection in detections:
            probs = detection[4:]
            index = np.argmax(probs)
            score = probs[index]
            if score >= self.model_config['conf_threshold']:
                if index != self.no_object_index:
                    x1 = int((detection[0] - detection[2]/2)*w)
                    y1 = int((detection[1] - detection[3]/2)*h)
                    x2 = int((detection[0] + detection[2]/2)*w)
                    y2 = int((detection[1] + detection[3]/2)*h)
                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
                    class_names.append(self.labels[index])
                
        boxes, scores, class_names = self.remove_overlapping_boxes(boxes, scores, class_names, threshold=0.4)
        return boxes, scores, class_names


    def predict(self, doc_info):
        doc_table_data = doc_info['tables']
        all_cells = []
        next_row_index = 0
        prev_max_row = -1
        for i, (page_index, page_img, table_data) in enumerate(zip(doc_info['indexes'], doc_info['images'], doc_table_data)):
            for table_index, table_img in enumerate(table_data['images']):
                boxes, scores, classes = self.predict_row_col(table_img)
                table_text_boxes = table_data['text_boxes'][table_index]
                boxes, scores, classes = self.correct_boxes(boxes, scores, classes, table_text_boxes)

                # # debug plot
                # tmp_img = table_img.copy()
                # for bb, score, cl in zip(boxes, scores, classes):
                #     bb = list(map(int, bb))
                #     if 'row' in cl:
                #         color = (0,0,255)
                #     elif 'col' in cl:
                #         color = (0,255,0)
                #     elif 'span' in cl:
                #         color = (255,0,0)
                #     else:
                #         continue
                #     cv2.rectangle(tmp_img, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                #     pos = (bb[0], bb[1]+10)
                #     cv2.putText(tmp_img, f'{score:.2f}', pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                # cv2.imwrite('test.jpg', tmp_img)
                # pdb.set_trace()


                ### 5.1
                rows, cols, spans = [], [], []
                for box, label in zip(boxes, classes):
                    if label == 'row':
                        rows.append(box)
                    elif label == 'col':
                        cols.append(box)
                    elif label == 'span':
                        spans.append(box)
                rows = sorted(rows, key=lambda x:x[1]+x[3])
                cols = sorted(cols, key=lambda x:x[0]+x[2])
                cells = self.extract_cells(rows, cols, spans)

                ### 5.2
                table_bbox = table_data['boxes'][table_index].tolist()
                cells = self.texts2cells(table_data['text'][table_index], cells, table_bbox, page_img.shape[:2])

                num_table_row = 0
                for cell_index, cell in enumerate(cells):
                    start_row, end_row, start_col, end_col = cell['relation']
                    num_table_row = max(num_table_row, end_row)
                    start_row, end_row = start_row + next_row_index, end_row + next_row_index
                    assert start_row > prev_max_row
                    cell['relation'] = [start_row, end_row, start_col, end_col]
                    cell['page_index'] = i
                if len(cells) > 0:
                    num_table_row += 1
                all_cells.extend(cells)
                next_row_index += num_table_row
                prev_max_row = max([cell['relation'][1] for cell in all_cells])

                # pdb.set_trace()

        doc_info['extracted_infos']['table_infos'] = all_cells
        return doc_info