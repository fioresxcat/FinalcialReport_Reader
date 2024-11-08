import pdb
import cv2
import numpy as np

from modules.object_detection.yolov8.base_object_detection import BaseObjectDetection, LetterBox


class TableDetector(BaseObjectDetection):
    instance = None
    
    def __init__(self, common_config, model_config):
        super().__init__(common_config, model_config)
        input_shape = model_config['input_shape']
        self.resizer = LetterBox((input_shape[0], input_shape[1]), auto=False)
        self.labels = ['table']
        
        
    @staticmethod
    def get_instance(common_config, model_config):
        if TableDetector.instance is None:
            TableDetector.instance = TableDetector(common_config, model_config)
        return TableDetector.instance
    
    
    def preprocess(self, image):
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #??? need or not
        image = self.resizer(image)
        # image = np.expand_dims(image[..., ::-1], 0).transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        image = image[..., ::-1].transpose((2, 0, 1))
        image = np.ascontiguousarray(image, dtype='float32')  # contiguous
        image /= 255  # 0 - 255 to 0.0 - 1.0
        return image
    
    
    def predict(self, request_id, inp, out, metadata):
        result = inp.get_data()
        batch_images = []
        batch_image_shapes = []
        result['pages'] = [{} for _ in range(len(result['refined_images']))]
        outputs = []
        if len(result['refined_images']) > 0:
            for refined_image in result['refined_images']:
                batch_image_shapes.append(refined_image.shape)
                processed_img = self.preprocess(refined_image)
                batch_images.append(processed_img)
                if len(batch_images) == self.model_config['max_batch_size']:
                    # batch_images =  np.concatenate(batch_images, axis=0)
                    batch_images = np.array(batch_images)
                    output_dict = self.request(batch_images)
                    outputs.extend(output_dict.as_numpy(self.model_config['output_name']))
                    metadata = self.add_metadata(metadata, 1, self.model_config['max_batch_size'])
                    batch_images = []

            if len(batch_images) > 0:
                # batch_images = np.concatenate(batch_images, axis=0)
                batch_images = np.array(batch_images)
                output_dict = self.request(batch_images)
                outputs.extend(output_dict.as_numpy(self.model_config['output_name']))

                metadata = self.add_metadata(metadata, 1, self.model_config['max_batch_size'])
                batch_images = []

            outputs = np.array(outputs)
            detections = self.non_max_suppression(
                outputs,
                conf_thres=self.model_config['conf_threshold'],
                iou=self.model_config['iou_threshold']
            )
            for i, detection in enumerate(detections):
                page_data = result['pages'][i]
                box_data = {
                    'boxes': [],
                    'scores': [],
                    'classes': [],
                    'images': []
                }
                if len(detection) != 0:
                    boxes, scores, class_ids = detection[:, :4], detection[:, 4], detection[:, 5]
                    boxes = self.scale_boxes((self.model_config['input_shape'][0], self.model_config['input_shape'][1]), boxes, (batch_image_shapes[i][0], batch_image_shapes[i][1]))
                    class_names = [self.labels[int(d)] for d in class_ids]
                    # Arange box from top to bottom #
                    indices = np.argsort(boxes[:, 3])
                    corrected_boxes = np.array([boxes[i] for i in indices]).astype(np.int32)
                    corrected_classes = [class_names[i] for i in indices]
                    corrected_scores = [scores[i] for i in indices]
                    for box in corrected_boxes:
                        x1, y1, x2, y2 = box
                        table_im = result['refined_images'][i][y1:y2, x1:x2]
                        box_data['images'].append(table_im)
                    box_data['boxes'] = corrected_boxes
                    box_data['scores'] = corrected_scores
                    box_data['classes'] = corrected_classes
                
                page_data['tables'] = box_data
                page_data['has_table'] = len(box_data['boxes']) > 0
                
                # # debug plot
                # for box, score, cl in zip(box_data['boxes'], box_data['scores'], box_data['classes']):
                #     x1, y1, x2, y2 = box
                #     temp_img = result['refined_images'][i]
                #     cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                #     cv2.imwrite('test.jpg', temp_img)
                #     pdb.set_trace()

        out.set_data(result)
        return out, metadata
