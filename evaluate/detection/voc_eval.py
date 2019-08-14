import numpy as np

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(filename, det_scores, det_labels, tp, fp, num_gt, use_07_metric=False):
    """ VOC mAP evaluate function
    :param det_bboxes: IxNx5 size, every cell is [x1 y1 x2 y2 score] numpy array
    :type det_bboxes:
    :param det_labels:
    :type det_labels:
    :param tp: true positive for det_bboxes, with IOU threshold 0.5
    :type tp:
    :param fp: false positive for det_bboxes, with IOU threshold 0.5
    :type fp:
    :return:
    :rtype:
    """
    filename_list = []
    for f in filename:
        filename_list += f
    if isinstance(det_scores, list):
        det_scores = np.concatenate(det_scores, axis=0)
    if isinstance(det_labels, list):
        det_labels = np.concatenate(det_labels, axis=0)
    if isinstance(tp, list):
        tp = np.concatenate(tp, axis=0)
    if isinstance(fp, list):
        fp = np.concatenate(fp, axis=0)

    class_labels = np.unique(det_labels)
    class_AP = []
    sorted_ind = np.argsort(-det_scores)
    tp = tp[sorted_ind]
    fp = fp[sorted_ind]
    det_labels = det_labels[sorted_ind]
    det_scores = det_scores[sorted_ind]
    filename_list = [filename_list[i] for i in sorted_ind]

    class_result = {}
    for c in class_labels:
        ind = (det_labels == c)
        class_tp = tp[ind]
        class_fp = fp[ind]

        class_result[c] = dict(
            class_tp=class_tp,
            class_fp=class_fp,
            det_scores=det_scores[ind],
            filename_list=[filename_list[i] for i in np.where(det_labels == c)[0]]
        )

        csum_tp = np.cumsum(class_tp, dtype=np.float32)
        csum_fp = np.cumsum(class_fp, dtype=np.float32)
        recall = csum_tp / num_gt[int(c)]
        precision = csum_tp / np.maximum((csum_tp + csum_fp), np.finfo(np.float64).eps)

        class_AP.append(voc_ap(recall, precision, use_07_metric))

    import torch
    torch.save(class_result, 'temp0.pth')

    return class_AP




