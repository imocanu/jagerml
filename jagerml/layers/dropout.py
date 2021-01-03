#!/usr/bin/env python3

from jagerml.helper import *


class Dropout:

    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binaryMask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binaryMask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binaryMask