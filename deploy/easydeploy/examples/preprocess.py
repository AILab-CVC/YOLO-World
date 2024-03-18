from typing import List, Tuple, Union

import cv2
import numpy as np
from config import ModelType
from numpy import ndarray


class Preprocess:

    def __init__(self, model_type: ModelType):
        if model_type in (ModelType.YOLOV5, ModelType.YOLOV6, ModelType.YOLOV7,
                          ModelType.YOLOV8):
            mean = np.array([0, 0, 0], dtype=np.float32)
            std = np.array([255, 255, 255], dtype=np.float32)
            is_rgb = True
        elif model_type == ModelType.YOLOX:
            mean = np.array([0, 0, 0], dtype=np.float32)
            std = np.array([1, 1, 1], dtype=np.float32)
            is_rgb = False
        elif model_type == ModelType.PPYOLOE:
            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            is_rgb = True

        elif model_type == ModelType.PPYOLOEP:
            mean = np.array([0, 0, 0], dtype=np.float32)
            std = np.array([255, 255, 255], dtype=np.float32)
            is_rgb = True
        elif model_type == ModelType.RTMDET:
            mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
            std = np.array([57.375, 57.12, 58.3955], dtype=np.float32)
            is_rgb = False
        else:
            raise NotImplementedError

        self.mean = mean.reshape((3, 1, 1))
        self.std = std.reshape((3, 1, 1))
        self.is_rgb = is_rgb

    def __call__(self,
                 image: ndarray,
                 new_size: Union[List[int], Tuple[int]] = (640, 640),
                 **kwargs) -> Tuple[ndarray, Tuple[float, float]]:
        # new_size: (height, width)
        height, width = image.shape[:2]
        ratio_h, ratio_w = new_size[0] / height, new_size[1] / width
        image = cv2.resize(
            image, (0, 0),
            fx=ratio_w,
            fy=ratio_h,
            interpolation=cv2.INTER_LINEAR)
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        image = image.astype(np.float32)
        image -= self.mean
        image /= self.std
        return image[np.newaxis], (ratio_w, ratio_h)
