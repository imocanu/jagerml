#!/usr/bin/env python3

from jagerml.helper import *


class LeakyReLU:

    def __init__(self, alpha=0.3):
        self.input = None
        self.output = None
        self.alpha = alpha

    def forward(self, inputs, training):
        self.input = inputs
        self.output = inputs.copy()
        self.output[inputs < 0] = self.output[inputs < 0] * self.alpha

    def backward(self, dvalues):
        self.dinputs = np.ones_like(dvalues)
        self.dinputs[dvalues < 0] *= self.alpha

    def predictions(self, outputs):
        return outputs
