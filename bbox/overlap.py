import torch
import numpy as np
from .transform import xywh2xyxy

# @profile
def bbox_overlap(bbox1, bbox2, xywh=True):
    """Computing iou between two bboxes.
        The input bbox in format xywh.
    """

    if isinstance(bbox1, torch.Tensor):
        if bbox1.size(0) == 0 or bbox2.size(0) == 0:
            return None

        # convert format to ltrb
        if xywh:
            bbox_corner1 = xywh2xyxy(bbox1)
            bbox_corner2 = xywh2xyxy(bbox2)
            area1 = bbox1[..., 2]*bbox1[..., 3]
            area2 = bbox2[..., 2]*bbox2[..., 3]
        else:
            bbox_corner1 = bbox1
            bbox_corner2 = bbox2
            area1 = (bbox1[..., 2]-bbox1[..., 0])*(bbox1[..., 3]-bbox1[..., 1])
            area2 = (bbox2[..., 2]-bbox2[..., 0])*(bbox2[..., 3]-bbox2[..., 1])

        w = torch.min(bbox_corner1[..., None, 2], bbox_corner2[..., 2])-torch.max(bbox_corner1[..., None, 0], bbox_corner2[..., 0])
        h = torch.min(bbox_corner1[..., None, 3], bbox_corner2[..., 3])-torch.max(bbox_corner1[..., None, 1], bbox_corner2[..., 1])
        area_overlap = w.clamp(0)*h.clamp(0)

        iou = area_overlap / (area1[..., None] + area2 - area_overlap)

        return iou

    elif isinstance(bbox1, np.ndarray):
        if bbox1.shape[0] == 0 or bbox2.shape[0] == 0:
            return None

        # convert format to ltrb
        if xywh:
            bbox_corner1 = xywh2xyxy(bbox1)
            bbox_corner2 = xywh2xyxy(bbox2)
            area1 = bbox1[..., 2]*bbox1[..., 3]
            area2 = bbox2[..., 2]*bbox2[..., 3]
        else:
            bbox_corner1 = bbox1
            bbox_corner2 = bbox2
            area1 = (bbox1[..., 2]-bbox1[..., 0])*(bbox1[..., 3]-bbox1[..., 1])
            area2 = (bbox2[..., 2]-bbox2[..., 0])*(bbox2[..., 3]-bbox2[..., 1])

        w = np.minimum(bbox_corner1[..., None, 2], bbox_corner2[..., 2])-np.maximum(bbox_corner1[..., None, 0], bbox_corner2[..., 0])
        h = np.minimum(bbox_corner1[..., None, 3], bbox_corner2[..., 3])-np.maximum(bbox_corner1[..., None, 1], bbox_corner2[..., 1])
        area_overlap = w.clip(0)*h.clip(0)

        iou = area_overlap / (area1[..., None] + area2 - area_overlap)

        return iou
    else:
        raise NotImplementedError
