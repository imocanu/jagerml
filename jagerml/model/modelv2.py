#!/usr/bin/env python3


from time import time
from collections import OrderedDict
from jagerml.helper import *
from jagerml.layers import Input, Flatten
from jagerml.activations import Softmax, SoftmaxLossCrossentropy
from jagerml.evaluate import LossCategoricalCrossentropy


class ModelV2:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def _init_params(self):
        pass

    def _build_encoder(self):
        """
        CNN encoder
        """
        self.encoder = OrderedDict()
        self.encoder["Flatten"] = Flatten(optimizer=self.optimizer)

    def _build_decoder(self):
        """
        CNN decoder
        """
    @property
    def parameters(self):
        return {}

    @property
    def hyperparameters(self):
        return {}

    @property
    def derived_variabled(self):
        return {}

    @property
    def gradients(self):
        return {}

    def _sample(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

    def update(self):
        pass

    def flush_gradients(self):
        pass

    def fit(self):
        pass

