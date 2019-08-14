from .transform import xywh2xyxy, xyxy2xywh
from torchvision.ops import nms as nms
import torch

def nms_wrapper(bboxes, scores, ignore=None, nms_iou_thr = 0.7, num_before=-1, num_after=-1):
    """
    Input bboxes array: batch_size x num x 4
    Output bboxes list: every item in list is nms result
    """
    if isinstance(bboxes, torch.Tensor):
        nms_bboxes = []
        nms_scores = []
        for b in range(len(bboxes)):
            ind = None if ignore is None else (ignore[b]==0).nonzero().view(-1)
            bbox_list = bboxes[b] if ignore is None else bboxes[b][ind]
            score_list = scores[b] if ignore is None else scores[b][ind]

            sorted_ind = (-score_list).sort()[1]
            bbox_list = bbox_list[sorted_ind[:num_before]]
            score_list = score_list[sorted_ind[:num_before]]

            left_ind = nms(xywh2xyxy(bbox_list), score_list, nms_iou_thr)
            left_ind = left_ind.sort()[0][:num_after]

            nms_bboxes.append(bbox_list[left_ind])
            nms_scores.append(score_list[left_ind])


        return nms_bboxes, nms_scores

    else:
        raise NotImplementedError
