from .transform import xywh2xyxy, xyxy2xywh
from torchvision.ops import nms as nms
import torch

def nms_wrapper(bboxes, scores, ignore=None, nms_iou_thr = 0.7):
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

            left_ind = nms(xywh2xyxy(bbox_list), score_list, nms_iou_thr)
            nms_bboxes.append(bbox_list[left_ind])
            nms_scores.append(score_list[left_ind])

            # _, sorted_ind = (-score_list).sort()
            # bbox_list = bbox_list[sorted_ind]
            # score_list = score_list[sorted_ind]
            # suppression = torch.zeros(size=(bbox_list.size(0), ), dtype=torch.long)

            # TODO: try to optimize the speed with balance of memory

            # slow but save gpu memory
            # for i in range(bbox_list.size(0)):
            #     if suppression[i] == 1: continue
            #     row_left_ind = (suppression[i+1:]==0).nonzero().view(-1) + (i+1)
            #     if row_left_ind.size(0) > 0:
            #         iou_row = bbox_overlap(bbox_list[i], bbox_list[row_left_ind])
            #         suppression_ind = row_left_ind[(iou_row >= nms_iou_thr).nonzero().view(-1)]
            #         suppression[suppression_ind] = 1

            # old version: fast but need more gpu memory
            # iou = bbox_overlap(bbox_list, bbox_list)
            # if iou is None:
            #     return None, None
            #
            # for i, iou_row in enumerate(iou):
            #     if suppression[i] == 1: continue
            #     if i != iou.size(0)-1:
            #         suppression[(iou_row[i+1:] > nms_iou_thr).nonzero() + (i+1)] = 1
            # left_ind = (suppression==0).nonzero().view(-1)


            # nms_bboxes.append(bbox_list[left_ind])
            # nms_scores.append(score_list[left_ind])

        return nms_bboxes, nms_scores

    else:
        raise NotImplementedError
