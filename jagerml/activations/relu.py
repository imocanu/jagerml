#!/usr/bin/env python3

from jagerml.imports import *

class ReLU:

    def __init__(self):
        pass

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <=0] = 0