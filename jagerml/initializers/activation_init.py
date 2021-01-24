#!/usr/bin/env python3

from jagerml.helper import *
from jagerml.activations import Linearbase, LeakyReLUbase


class ActivationInitializer:
    def __init__(self, param=None):
        self.param = param

    def __call__(self):
        param = self.param
        if param is None:
            print("Default is Linear activation !")
            act = Linearbase()
        elif isinstance(param, str):
            act = self.parse_str(param)
            act = param
        else:
            print("!!!! else !!!")
            act = param
        return act

    def parse_str(self, param):
        act_str = param.lower()
        if act_str == "LeakyReLUbase":
            print("LeakyReLUbase selected ..")
        else:
            print("NONE selected ..")

        return act_str
