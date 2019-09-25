from torch.nn.utils import clip_grad

from .hook import Hook


class OptimizerHook(Hook):

    def __init__(self, interval=1, grad_clip=None):
        self.interval=interval
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip)

    def after_train_iter(self, runner):
        runner.outputs['loss'].backward()

        if self.every_n_inner_iters(runner, self.interval):
            if self.grad_clip is not None:
                self.clip_grads(runner.model.parameters())
            runner.optimizer.step()
            runner.optimizer.zero_grad()
