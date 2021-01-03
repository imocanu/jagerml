#!/usr/bin/env python3

from jagerml.helper import *


class Dense:
    def __init__(self, nInputs, nNeurons,
                 weightL1=0, biasL1=0,
                 weightL2=0, biasL2=0):

        self.weights = 0.1 * np.random.randn(nInputs, nNeurons)
        self.biases = np.zeros((1, nNeurons))
        self.weightL1 = weightL1
        self.weightL2 = weightL2
        self.biasL1 = biasL1
        self.biasL2 = biasL2

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases  = np.sum(dvalues, axis=0, keepdims=True)

        if self.weightL1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weightL1 * dL1

        if self.weightL2 > 0:
            self.dweights += 2 * self.weightL2 * self.weights

        if self.biasL1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.biasL1 * dL1

        if self.biasL2 > 0:
            self.dbiases += 2 * self.biasL2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)
