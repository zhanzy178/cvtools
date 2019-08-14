import numpy as np
from cvtools.bbox import bbox_overlap

def tpfp(det_bboxes, det_labels, gt_bboxes, gt_labels, difficults, class_num, iou_thr=0.5):
    num_gt = np.histogram(gt_labels, np.arange(0, class_num + 1))[0]
    gt_matched = np.zeros(gt_bboxes.shape[0])
    tp = np.zeros(det_bboxes.shape[0])
    fp = np.zeros(det_bboxes.shape[0])

    for di in range(det_bboxes.shape[0]):
        if not num_gt[det_labels[di]]:
            fp[di] = 1
            continue
        cls_ind = np.where(gt_labels == det_labels[di])[0]
        iou = bbox_overlap(det_bboxes[di], gt_bboxes[cls_ind])

        max_ind = np.argmax(iou)
        gt_ind = cls_ind[max_ind]
        if iou[max_ind] > iou_thr:
            # same class
            if not difficults[gt_ind]:
                if gt_matched[gt_ind] == 0:
                    tp[di] = 1
                    gt_matched[gt_ind] = 1
                else:
                    fp[di] = 1
        else:
            fp[di] = 1

    return tp, fp, num_gt
