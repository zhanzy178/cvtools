import numpy as np


def tpfp(det_scores, det_labels, gt_bboxes, gt_labels, iou_mat, iou_thr=0.5):
    gt_matched = np.zeros(gt_bboxes.shape[0])
    tp = np.zeros(det_scores.shape[0])
    fp = np.zeros(det_scores.shape[0])

    for iou, di in enumerate(iou_mat):
        max_ind = np.argmax(iou)

        if iou[max_ind] < iou_thr:
            fp[di] = 1
        else:
            if gt_matched[max_ind] == 0:
                if det_labels[di] == gt_labels[max_ind]:
                    tp[di] = 1
                    gt_matched[max_ind] = 1
                else:
                    fp[di] = 1

    return tp, fp
