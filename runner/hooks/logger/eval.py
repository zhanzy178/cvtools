from ..hook import Hook
from cvtools.evaluate import tpfp, voc_eval
from cvtools.bbox import bbox_overlap
from collections import OrderedDict
import numpy as np
import torch
import datetime
import torch.distributed as dist

class EvalLoggerHook(Hook):
    def __init__(self, interval=100):
        super(EvalLoggerHook, self).__init__()
        self.interval = interval
        self.time_sec_tot = 0

    def _get_max_memory(self, runner):
        mem = torch.cuda.max_memory_allocated()
        mem_mb = torch.tensor([mem / (1024 * 1024)],
                              dtype=torch.int,
                              device=torch.device('cuda'))
        if runner.world_size > 1:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()

    def before_val_epoch(self, runner):
        self.start_iter = 0
        self.time_sec_tot = 0
        runner.log_buffer.clear()

    def after_val_epoch(self, runner):
        self.eval(runner)
        self.after_val_epoch_log(runner)
        runner.log_buffer.clear()

    def after_val_iter(self, runner):
        # compute tp fp
        det_bboxes = runner.outputs['det_bboxes']
        det_labels = runner.outputs['det_labels']
        gt_bboxes = runner.outputs['gt_bboxes']
        gt_labels = runner.outputs['gt_labels']

        for i in range(len(det_bboxes)):
            iou_mat = bbox_overlap(det_bboxes[i], gt_bboxes[i], xywh=True)
            if iou_mat is None: continue
            tp, fp = tpfp(det_bboxes[i][:, -1], det_labels[i], gt_bboxes[i], gt_labels[i], iou_mat)
            runner.log_buffer.update(dict(
                scores=det_bboxes[i][:, -1],
                labels=det_labels[i],
                tp=tp,
                fp=fp,
                num_gt=np.histogram(gt_labels, np.arange(1, 22))[0]
            ))

        # log with interval
        if self.every_n_inner_iters(runner, self.interval):
            runner.log_buffer.average(keys=['time', 'data_time'])
            self.text_log(runner)

    def _log_info(self, log_dict, runner):
        log_str = 'Epoch [{}][{}/{}]\t'.format(
            log_dict['epoch'], log_dict['iter'], len(runner.data_loader))
        if 'time' in log_dict.keys():
            self.time_sec_tot += (log_dict['time'] * self.interval)
            time_sec_avg = self.time_sec_tot / (
                runner.inner_iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (runner.max_iters - runner.inner_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            log_str += 'val epoch eta: {}, '.format(eta_str)
            log_str += ('time: {:.3f}, data_time: {:.3f}, '.format(
                log_dict['time'], log_dict['data_time']))
            log_str += 'memory: {}, '.format(log_dict['memory'])

        runner.logger.info(log_str)

    def text_log(self, runner):
        log_dict = OrderedDict()
        log_dict['mode'] = 'val'
        log_dict['epoch'] = runner.epoch
        log_dict['iter'] = runner.inner_iter + 1
        log_dict['time'] = runner.log_buffer.output['time']
        log_dict['data_time'] = runner.log_buffer.output['data_time']
        if torch.cuda.is_available():
            log_dict['memory'] = self._get_max_memory(runner)

        self._log_info(log_dict, runner)


    def eval(self, runner):
        pass

    def after_val_epoch_log(self, runner):
        pass



class VOCEvalLoggerHook(EvalLoggerHook):
    def eval(self, runner):
        scores = runner.log_buffer.val_history['scores']
        labels = runner.log_buffer.val_history['labels']
        tp = runner.log_buffer.val_history['tp']
        fp = runner.log_buffer.val_history['fp']
        num_gt = np.sum(runner.log_buffer.val_history['num_gt'], axis=0)

        mAP = voc_eval(scores, labels, tp, fp, num_gt)
        runner.log_buffer.update(dict(mAP=mAP))

    def after_val_epoch_log(self, runner):
        runner.log_buffer.average(keys=['mAP'])
        log_items = []
        log_items.append('{}: {:.4f}'.format('mAP', runner.log_buffer.output['mAP']))
        log_str = ', '.join(log_items)

        runner.logger.info(log_str)
