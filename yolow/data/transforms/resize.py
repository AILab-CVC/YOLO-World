import cv2
import numpy as np
from typing import Tuple, Union

__all__ = (
    'YOLOResize',
)


class YOLOResize:

    def __init__(
            self,
            scale: Union[int, Tuple[int, int]]):
        assert scale is not None
        if isinstance(scale, int):
            self.scale = (scale, scale)
        else:
            self.scale = scale

    def resize_image(self, results: dict):
        image = results.get('img', None)
        if image is None:
            return

        if 'batch_shape' in results:
            input_size = tuple(results['batch_shape'])  # hw
        else:
            input_size = self.scale[::-1]  # wh -> hw

        if len(image.shape) == 3:
            padded_image = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_image = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / image.shape[0], input_size[1] / image.shape[1])
        resized_image = cv2.resize(
            image,
            (int(image.shape[1] * r), int(image.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_image[: int(image.shape[0] * r), : int(image.shape[1] * r)] = resized_image
        padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)

        scale_factor = (r, r)
        pad_param = np.array([0, input_size[0] - int(image.shape[0] * r),
                              0, input_size[1] - int(image.shape[1] * r)],
                             dtype=np.float32)

        results['img'] = padded_image
        results['img_shape'] = padded_image.shape
        results['scale_factor'] = scale_factor
        results['pad_param'] = pad_param

    def resize_bboxes(self, results: dict):
        if results.get('gt_bboxes', None) is None:
            return
        scale_factor = results['gt_bboxes'].new_tensor(results['scale_factor']).repeat(2)
        results['gt_bboxes'] *= scale_factor

        if len(results['pad_param']) != 4:
            return
        results['gt_bboxes'] += results['gt_bboxes'].new_tensor(
            (results['pad_param'][2], results['pad_param'][0])).repeat(2)

        if self.clip_object_border:
            results['gt_bboxes'][..., 0::2] = results['gt_bboxes'][..., 0::2].clamp(0, results['img_shape'][1])
            results['gt_bboxes'][..., 1::2] = results['gt_bboxes'][..., 1::2].clamp(0, results['img_shape'][0])

    def transform(self, results: dict) -> dict:
        results['scale'] = self.scale
        self.resize_image(results)
        self.resize_bboxes(results)
        return results

    def __call__(self, results: dict) -> dict:
        results = self.transform(results)
        return results
