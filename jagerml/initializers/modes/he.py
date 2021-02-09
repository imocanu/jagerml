#!/usr/bin/env python3

from jagerml.helper import *
from jagerml.initializers.modes import calc_fan, truncated


def he_uniform(weight_shape):
    fan_in, fan_out = calc_fan(weight_shape)
    b = np.sqrt(6 / fan_in)
    return np.random.uniform(-b, b, size=weight_shape)


def he_normal(weight_shape):
    fan_in, fan_out = calc_fan(weight_shape)
    b = np.sqrt(2 / fan_in)
    return truncated(0, b, weight_shape)

