#!/usr/bin/env python3

from jagerml.helper import *
from jagerml.optimizers import OptimizerBase, SGDbase


class OptimizerInitializer:
    def __init__(self, param=None):
        self.param = param

    def __call__(self):
        param = self.param
        if param is None:
            opt = SGDbase()
        elif isinstance(param, OptimizerBase):
            opt = param
        else:
            raise ValueError("Optimizer not found !")
        return opt
