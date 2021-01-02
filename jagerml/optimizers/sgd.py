#!/usr/bin/env python3

from jagerml.imports import *

class SGD:
    def __init__(self, learningRate=1., decay=0., momentum=0.):
        self.learningRate = learningRate
        self.currentlearningRate = learningRate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def preUpdateParams(self):
        if self.decay:
            self.currentlearningRate = self.learningRate * \
                                       ( 1. / (1. + self.decay * self.iterations))

    def updateParams(self, layer):
        # layer.weights += -self.currentlearningRate * layer.dweights
        # layer.biases  += -self.currentlearningRate * layer.dbiases
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

                weightUpdates = self.momentum * layer.weight_momentums - self.currentlearningRate * layer.dweights
                layer.weight_momentums = weightUpdates

                biasUpdates = self.momentum * layer.bias_momentums - self.currentlearningRate * layer.dbiases
                layer.bias_momentums = biasUpdates
            else:
                weightUpdates = -self.currentlearningRate * layer.dweights
                biasUpdates = -self.currentlearningRate * layer.dbiases

            layer.weights += weightUpdates
            layer.biases += biasUpdates

    def postUpdateParams(self):
        self.iterations += 1