#!/usr/bin/env python3

from jagerml.imports import *

class AdaGrad:
    def __init__(self, learningRate=0.001, decay=0., momentum=0., epsilon=1e-7):
        self.learningRate = learningRate
        self.currentlearningRate = learningRate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        self.epsilon = epsilon

    def preUpdateParams(self):
        if self.decay:
            self.currentlearningRate = self.learningRate * \
                                       ( 1. / (1. + self.decay * self.iterations))

    def updateParams(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.currentlearningRate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.currentlearningRate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def postUpdateParams(self):
        self.iterations += 1