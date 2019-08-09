from ..hook import Hook
from .text import TextLoggerHook
from cvtools.evaluate import tpfp, voc_eval
from cvtools.bbox import bbox_overlap
import numpy as np

class EvalLoggerHook(TextLoggerHook):
    def __init__(self):
        super(EvalLoggerHook, self).__init__()

    def before_val_epoch(self, runner):
        runner.log_buffer.clear()

    def after_val_iter(self, runner):
        det_bboxes = runner.outputs['det_bboxes']
        det_labels = runner.outputs['det_labels']
        gt_bboxes = runner.outputs['gt_bboxes']
        gt_labels = runner.outputs['gt_labels']

        for i in range(len(det_bboxes)):
            iou_mat = bbox_overlap(det_bboxes[i], gt_bboxes[i], xywh=True)
            tp, fp = tpfp(det_bboxes[i][:, -1], det_labels[i], gt_bboxes[i], gt_labels[i], iou_mat)
            runner.log_buffer.update(dict(
                scores=det_bboxes[i][:, -1],
                labels=det_labels[i],
                tp=tp,
                fp=fp,
                num_gt=np.histogram(gt_labels, np.arange(1, 22))
            ))

    def eval(self, runner):
        pass

    def after_val_epoch(self, runner):
        self.eval(runner)
        runner.log_buffer.clear()



class VOCEvalLoggerHook(EvalLoggerHook):
    def eval(self, runner):
        scores = runner.log_buffer.history_val['scores']
        labels = runner.log_buffer.history_val['labels']
        tp = runner.log_buffer.history_val['tp']
        fp = runner.log_buffer.history_val['fp']
        num_gt = np.sum(runner.log_buffer.history_val['num_gt'], axis=0)

        mAP = voc_eval(scores, labels, tp, fp, num_gt)
