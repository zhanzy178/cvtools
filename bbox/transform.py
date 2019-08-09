import torch
import numpy as np

def xywh2xyxy(bbox_t):
    """This function maps feature map bbox xywh format to x1y1x2y2 format."""
    if isinstance(bbox_t, torch.Tensor):
        bbox = bbox_t.clone()
    elif isinstance(bbox_t, np.array):
        bbox = bbox_t.copy()
    else:
        raise NotImplementedError

    bbox[..., [0, 1]] -= bbox[..., [2, 3]] / 2
    bbox[..., [2, 3]] += bbox[..., [0, 1]]
    return bbox


def xyxy2xywh(bbox_t):
    """This function maps feature map bbox x1y1x2y2 format to xywh format."""
    if isinstance(bbox_t, torch.Tensor):
        bbox = bbox_t.clone()
    elif isinstance(bbox_t, np.array):
        bbox = bbox_t.copy()
    else:
        raise NotImplementedError

    bbox[..., [2, 3]] = bbox[..., [2, 3]] - bbox[:, [0, 1]]
    bbox[..., [0, 1]] += bbox[..., [2, 3]] / 2
    return bbox
