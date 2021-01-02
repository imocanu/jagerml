#!/usr/bin/env python3

from jagerml.imports import *
from jagerml.helper import *

class Dense:
    def __init__(self, nInputs, nNeurons):
        self.weights = np.random.randn(nInputs, nNeurons)
        self.biases  = np.zeros((1, nNeurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
        return True

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases  = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs  = np.dot(dvalues, self.weights.T)
