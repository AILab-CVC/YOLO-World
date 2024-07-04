# Modify from https://github.com/open-mmlab/mmcv/tree/main
# Apache-2.0 license
import cv2
import numpy as np
import pycocotools.mask as maskUtils
import torch
from cv2 import IMREAD_COLOR, IMREAD_UNCHANGED
from typing import Optional, Union

__all__ = (
    'LoadImageFromFile',
    'LoadAnnotations',
)


class LoadImageFromFile:
    """Load an image from file.
    """

    def __init__(self, to_float32: bool = False) -> None:
        self.to_float32 = to_float32
        # ONLY CV2
        self.imdecode_backend = 'cv2'

    def __call__(self, results: dict) -> Optional[dict]:
        filename = results['img_path']
        try:
            with open(filename, 'rb') as f:
                img_bytes = f.read()
                img_np = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img_np, IMREAD_COLOR)  # bgr
        except Exception as e:
            raise e
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"imdecode_backend='{self.imdecode_backend}', ")
        return repr_str


class LoadAnnotations:
    """Load and process the ``instances`` and ``seg_map`` annotation provided
    by dataset.

    - gt_bboxes (np.float32)
    - gt_bboxes_labels (np.int64)
    - gt_seg_map (np.uint8)
    - gt_keypoints (np.float32)
    """

    def __init__(
        self,
        with_bbox: bool = True,
        with_label: bool = True,
        with_seg: bool = False,
        with_mask: bool = False,
        poly2mask: bool = True,
    ) -> None:
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_seg = with_seg
        self.with_mask = with_mask
        self.poly2mask = poly2mask
        self.imdecode_backend = 'cv2'

    def _load_bboxes(self, results: dict) -> None:
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_bboxes.append(instance['bbox'])
            gt_ignore_flags.append(instance['ignore_flag'])
        results['gt_bboxes'] = np.array(gt_bboxes, dtype=np.float32).reshape((-1, 4))
        results['gt_bboxes'] = torch.as_tensor(results['gt_bboxes'])
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

    def _load_labels(self, results: dict) -> None:
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            gt_bboxes_labels.append(instance['bbox_label'])
        results['gt_bboxes_labels'] = np.array(gt_bboxes_labels, dtype=np.int64)

    def _poly2mask(self, mask_ann: Union[list, dict], img_h: int, img_w: int) -> np.ndarray:
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _process_masks(self, results: dict) -> list:
        gt_masks = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_mask = instance['mask']
            # If the annotation of segmentation mask is invalid,
            # ignore the whole instance.
            if isinstance(gt_mask, list):
                gt_mask = [np.array(polygon) for polygon in gt_mask if len(polygon) % 2 == 0 and len(polygon) >= 6]
                if len(gt_mask) == 0:
                    # ignore this instance and set gt_mask to a fake mask
                    instance['ignore_flag'] = 1
                    gt_mask = [np.zeros(6)]
            elif not self.poly2mask:
                # `PolygonMasks` requires a ploygon of format List[np.array],
                # other formats are invalid.
                instance['ignore_flag'] = 1
                gt_mask = [np.zeros(6)]
            elif isinstance(gt_mask, dict) and \
                    not (gt_mask.get('counts') is not None and
                         gt_mask.get('size') is not None and
                         isinstance(gt_mask['counts'], (list, str))):
                # if gt_mask is a dict, it should include `counts` and `size`,
                # so that `BitmapMasks` can uncompressed RLE
                instance['ignore_flag'] = 1
                gt_mask = [np.zeros(6)]
            gt_masks.append(gt_mask)
            # re-process gt_ignore_flags
            gt_ignore_flags.append(instance['ignore_flag'])
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)
        return gt_masks

    def _load_masks(self, results: dict) -> None:
        # TODO mask type
        h, w = results['ori_shape']
        gt_masks = self._process_masks(results)
        if self.poly2mask:
            gt_masks = np.stack([self._poly2mask(mask, h, w) for mask in gt_masks]).reshape(-1, h, w)
        else:
            # fake polygon masks will be ignored in `PackDetInputs`
            gt_masks = ([mask for mask in gt_masks], h, w)
        results['gt_masks'] = gt_masks

    def _load_seg_map(self, results: dict) -> None:
        if results.get('seg_map_path', None) is None:
            return

        f = open(results['seg_map_path'], 'rb')
        img_bytes = f.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        gt_semantic_seg = cv2.imdecode(img_np, IMREAD_UNCHANGED).squeeze()

        # modify if custom classes
        # TODO check if can be deleted
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg

    def __call__(self, results: dict) -> dict:
        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_seg:
            self._load_seg_map(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        return repr_str
