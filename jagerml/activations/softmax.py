#!/usr/bin/env python3

from jagerml.imports import *

class Softmax:

    def __init__(self):
        pass

    def forward(self, inputs):
        expVals = np.exp(inputs - np.max(inputs,
                                        axis=1,
                                        keepdims=True))
        # Normalize
        normalize = expVals / np.sum(expVals,
                                     axis=1,
                                     keepdims=True)
        self.output = normalize