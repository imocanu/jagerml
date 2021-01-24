#!/usr/bin/env python3

from jagerml.helper import *


def calc_fan(weight_shape):
    if len(weight_shape) == 2:
        fan_in, fan_out = weight_shape
    elif len(weight_shape) in [3, 4]:
        in_ch, out_ch = weight_shape[-2:]
        kernel_size = np.prod(weight_shape[:-2])
        fan_in = in_ch * kernel_size
        fan_out = out_ch * kernel_size
    else:
        raise ValueError("weight dim error !")

    return fan_in, fan_out
