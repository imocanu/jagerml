#!/usr/bin/env python3

from jagerml.helper import *
from jagerml.initializers.modes import he_uniform, he_normal, glorot_uniform, glorot_normal


class WeightInitializer(ABC):
    def __init__(self, act_fn_str, mode="glorot_uniform"):
        self.mode = mode
        self.act_fn_str = act_fn_str

        if mode not in [
            "glorot_uniform",
            "glorot_normal",
            "he_uniform",
            "he_normal",
        ]:
            raise ValueError("Mode {} not found !!!".format(mode))

        if mode == "glorot_uniform":
            print("[debug] __init__ select glorot_uniform")
            self._fn = glorot_uniform
        elif mode == "glorot_normal":
            print("[debug] __init__ select glorot_normal")
            self._fn = glorot_normal
        elif mode == "he_uniform":
            print("[debug] __init__ select he_uniform")
            self._fn = he_uniform
        elif mode == "he_normal":
            print("[debug] __init__ select he_normal")
            self._fn = he_normal
        else:
            raise ValueError("error : unknown mode !")

    def __call__(self, weight_shape):
        if "glorot" in self.mode:
            gain = self._calc_glorot_gain()
            W = self._fn(weight_shape, gain)
        elif self.mode == "std_normal":
            W = self._fn(*weight_shape)
        else:
            W = self._fn(weight_shape)
        return W

    def _calc_glorot_gain(self):
        gain = 1.0
        act_str = self.act_fn_str.lower()
        if act_str == "relu":
            gain = np.sqrt(2)
        elif act_str == "leaky_relu":
            alpha = 0.3
            gain = np.sqrt(2 / 1 + float(alpha) ** 2)
        return gain
