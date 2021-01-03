#!/usr/bin/env python3

from jagerml.helper import *
from jagerml.evaluate import LossCategoricalCrossentropy


class Softmax:

    def forward(self, inputs):
        self.inputs = inputs
        expVals = np.exp(inputs - np.max(inputs,
                                         axis=1,
                                         keepdims=True))
        # normalize
        normalize = expVals / np.sum(expVals,
                                     axis=1,
                                     keepdims=True)
        self.output = normalize

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (singleOutput, singleDvalues) in enumerate(zip(self.output, dvalues)):
            singleOutput = singleOutput.reshape(-1, 1)
            jacobianMatrix = np.diagflat(singleOutput) - np.dot(singleOutput, singleOutput.T)
            self.dinputs[index] = np.dot(jacobianMatrix, singleDvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


class SoftmaxLossCrossentropy:
    def __init__(self):
        self.activation = Softmax()
        self.loss = LossCategoricalCrossentropy()

    def forward(self, inputs, yTrue):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, yTrue)

    def backward(self, dvalues, yTrue):
        samples = len(dvalues)

        if len(yTrue.shape) == 2:
            yTrue = np.argmax(yTrue, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), yTrue] -= 1
        self.dinputs = self.dinputs / samples

