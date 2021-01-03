#!/usr/bin/env python3

from jagerml.helper import *


class RMSprop:
    def __init__(self, learningRate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learningRate = learningRate
        self.currentlearningRate = learningRate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def preUpdateParams(self):
        if self.decay:
            self.currentlearningRate = self.learningRate * \
                                       ( 1. / (1. + self.decay * self.iterations))

    def updateParams(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weightCache = np.zeros_like(layer.weights)
            layer.biasCache = np.zeros_like(layer.biases)

        layer.weightCache += self.rho * layer.weightCache + (1 - self.rho) * layer.dweights ** 2
        layer.biasCache += self.rho * layer.biasCache + (1 - self.rho) * layer.dbiases ** 2

        layer.weights += -self.currentlearningRate * layer.dweights / (np.sqrt(layer.weightCache) + self.epsilon)
        layer.biases += -self.currentlearningRate * layer.dbiases / (np.sqrt(layer.biasCache) + self.epsilon)

    def postUpdateParams(self):
        self.iterations += 1