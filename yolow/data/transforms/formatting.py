# Modify from https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/transforms/formatting.py
# Apache-2.0 license
import numpy as np

from ..misc import to_tensor

__all__ = ('PackDetInputs', )


class PackDetInputs:
    """
    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_id``: id of the image

        - ``img_path``: path to the image file

        - ``ori_shape``: original shape of the image as a tuple (h, w)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction
    """
    mapping_table = {'gt_bboxes': 'bboxes', 'gt_bboxes_labels': 'labels', 'gt_masks': 'masks'}

    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def __call__(self, results: dict) -> dict:
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()

            packed_results['inputs'] = img

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        data_sample = dict()
        instance_data = dict()
        ignore_instance_data = dict()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or 'box' in key:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(results[key])
        data_sample['gt_instances'] = instance_data
        data_sample['ignored_instances'] = ignore_instance_data

        if 'proposals' in results:
            data_sample['proposals'] = dict(
                bboxes=to_tensor(results['proposals']), scores=to_tensor(results['proposals_scores']))

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            if 'ignore_index' in results:
                metainfo = dict(ignore_index=results['ignore_index'])
                gt_sem_seg_data['img_metas'] = metainfo
            data_sample['gt_sem_seg'] = gt_sem_seg_data

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample['img_metas'] = img_meta
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
