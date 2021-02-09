#!/usr/bin/env python3

from jagerml.helper import *
from jagerml.initializers.modes import calc_fan, truncated


def glorot_uniform(weight_shape, gain=1.0):
    print("[debug] call : glorot_uniform", weight_shape)
    fan_in, fan_out = 1, 1  # calc_fan(weight_shape)
    b = gain * np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-b, b, size=weight_shape)


def glorot_normal(weight_shape, gain=1.0):
    fan_in, fan_out = calc_fan(weight_shape=weight_shape)
    b = gain * np.sqrt(2 / (fan_in + fan_out))
    return truncated(0, b, weight_shape)
