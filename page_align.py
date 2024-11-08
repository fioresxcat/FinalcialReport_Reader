import pdb
import cv2
import numpy as np

from imutils import rotate_bound
from modules.base import BaseModule

class PageAligner(BaseModule):
    
    instance = None
    
    def __init__(self, common_config, model_config):
        self.common_config = common_config
        self.model_config = model_config
        
    
    @staticmethod
    def get_instance(common_config, model_config):
        if PageAligner.instance is None:
            PageAligner.instance = PageAligner(common_config, model_config)
        return PageAligner.instance


    def fast_rotate(self, arr, angle):
        # Get image dimensions
        h, w = arr.shape[:2]

        # Calculate the center of the image
        center = (w / 2, h / 2)

        # Compute the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform the rotation using nearest-neighbor interpolation
        rotated = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_NEAREST)

        return rotated

    def determine_score(self, arr, angle):
        data = self.fast_rotate(arr, angle)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score


    def get_deskew_angle(self, image):
        h, w = image.shape[:2]
        image = cv2.resize(image, (w//2, h//2))  # downscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        scores = []
        angles = np.arange(-2, 2.5, 0.5)   # 9 times
        for angle in angles:
            histogram, score = self.determine_score(thresh, angle)
            scores.append(score)
        best_angle = angles[scores.index(max(scores))]
        return best_angle

    def predict(self, request_id, inp, out, metadata):
        result = inp.get_data()
        result.pop('images', None)
        result['refined_images'] = []
        result['refined_images_shape'] = []
        for index, image in enumerate(result['rotated_images']):
            refined_angle = self.get_deskew_angle(image)
            refined_image = rotate_bound(image, -refined_angle)
            result['refined_images'].append(refined_image)
            result['refined_images_shape'].append(refined_image.shape[:2])
            # result['rotated_images'][index] = None
        metadata = self.add_metadata(metadata, 1, 1)

        result.pop('rotated_images', None)
        out.set_data(result)
        return out, metadata
        
