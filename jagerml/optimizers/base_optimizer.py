#!/usr/bin/env python3

from jagerml.helper import *
from numpy.linalg import norm


class OptimizerBase(ABC):
    def __init__(self, lr, scheduler=None):
        self.cache = {}
        self.cur_step = 0
        self.hyperparameters = {}
        self.lr_scheduler = lr
        self.scheduler = scheduler

    def __call__(self, param, param_grad, param_name, cur_loss=None):
        return self.update(param, param_grad, param_name, cur_loss)

    def step(self):
        """Increment the optimizer step counter by 1"""
        self.cur_step += 1

    def reset_step(self):
        """Reset the step counter to zero"""
        self.cur_step = 0

    # def copy(self):
    #     """Return a copy of the optimizer object"""
    #     return deepcopy(self)

    @abstractmethod
    def update(self, param, param_grad, param_name, cur_loss=None):
        raise NotImplementedError


class SGDbase(OptimizerBase):
    def __init__(self, lr=0.01, momentum=0.0, clip_norm=None, lr_scheduler=None):
        super().__init__(lr, scheduler=lr_scheduler)

        self.hyperparameters = {
            "id": "SGD",
            "lr": lr,
            "momentum": momentum,
            "clip_norm": clip_norm,
            "ls_scheduler": str(self.lr_scheduler)
        }

    def __str__(self):
        H = self.hyperparameters
        lr, mm, cn, sc = H["lr"], H["momentum"], H["clip_norm"], H["lr_scheduler"]
        return "SGD(lr={}, momentum={}, clip_norm={}, lr_scheduler={})".format(
            lr, mm, cn, sc
        )

    def update(self, param, param_grad, param_name, cur_loss=None):
        C = self.cache
        H = self.hyperparameters
        momentum, clip_norm = H["momentum"], H["clip_norm"]
        lr = self.lr_scheduler(self.cur_step, cur_loss)

        if param_name not in C:
            C[param_name] = np.zeros_like(param_grad)

        # scale gradient to avoid explosion
        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        update = momentum * C[param_name] + lr * param_grad
        self.cache[param_name] = update
        return param - update
