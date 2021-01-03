#!/usr/bin/env python3

from jagerml.helper import *


class ReLU:

    def forward(self, inputs, training):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs