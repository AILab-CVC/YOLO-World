import numpy as np
import torch
from collections import abc
from typing import Any, Mapping, Sequence, Type, Union

from ..dist_utils import get_dist_info, sync_random_seed

__all__ = ('get_dist_info', 'sync_random_seed', 'to_tensor', 'pseudo_collate', 'imflip', 'is_list_of')


def to_tensor(data: Union[torch.Tensor, np.ndarray, Sequence, int, float]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


def pseudo_collate(data_batch: Sequence) -> Any:
    """Convert list of data sampled from dataset into a batch of data, of which
    type consistent with the type of each data_itement in ``data_batch``.

    The default behavior of dataloader is to merge a list of samples to form
    a mini-batch of Tensor(s). However, in MMEngine, ``pseudo_collate``
    will not stack tensors to batch tensors, and convert int, float, ndarray to
    tensors.

    This code is referenced from:
    `Pytorch default_collate <https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py>`_.

    Args:
        data_batch (Sequence): Batch of data from dataloader.

    Returns:
        Any: Transversed Data in the same format as the data_itement of
        ``data_batch``.
    """  # noqa: E501
    data_item = data_batch[0]
    data_item_type = type(data_item)
    if isinstance(data_item, (str, bytes)):
        return data_batch
    elif isinstance(data_item, tuple) and hasattr(data_item, '_fields'):
        # named tuple
        return data_item_type(*(pseudo_collate(samples) for samples in zip(*data_batch)))
    elif isinstance(data_item, Sequence):
        # check to make sure that the data_itements in batch have
        # consistent size
        it = iter(data_batch)
        data_item_size = len(next(it))
        if not all(len(data_item) == data_item_size for data_item in it):
            raise RuntimeError('each data_itement in list of batch should be of equal size')
        transposed = list(zip(*data_batch))

        if isinstance(data_item, tuple):
            return [pseudo_collate(samples) for samples in transposed]  # Compat with Pytorch.
        else:
            try:
                return data_item_type([pseudo_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)`
                # (e.g., `range`).
                return [pseudo_collate(samples) for samples in transposed]
    elif isinstance(data_item, Mapping):
        return data_item_type({
            key:
            pseudo_collate([d[key] for d in data_batch]) if key != 'data_samples' else [d[key] for d in data_batch]
            for key in data_item
        })
    else:
        return data_batch


def imflip(img: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
    """Flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray: The flipped image.
    """
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction == 'horizontal':
        return np.flip(img, axis=1)
    elif direction == 'vertical':
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))


def is_seq_of(seq: Any, expected_type: Union[Type, tuple], seq_type: Type = None) -> bool:
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type or tuple): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.

    Examples:
        >>> from mmengine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)
