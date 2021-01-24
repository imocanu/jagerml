#!/usr/bin/env python3
from abc import ABC

from jagerml.helper import *
from jagerml.initializers.modes import calc_fan


class HeUniform(ABC):
    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed
        self.scale = 1.0
        self.mode = "fan_in"
        self.distribution = "truncated_normal"

    def get_config(self):
        return {'seed', self.seed}

    def __call__(self, weight_shape):
        fan_in, fan_out = calc_fan(weight_shape)
        b = np.sqrt(6 / fan_in)
        return np.random.uniform(-b, b, size=weight_shape)
