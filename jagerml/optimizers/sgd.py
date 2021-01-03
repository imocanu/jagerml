#!/usr/bin/env python3

from jagerml.helper import *


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
                                       (1. / (1. + self.decay * self.iterations))

    def updateParams(self, layer):
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weightMomentums = np.zeros_like(layer.weights)
                layer.biasMomentums = np.zeros_like(layer.biases)

                weightUpdates = self.momentum * layer.weightMomentums - self.currentlearningRate * layer.dweights
                layer.weightMomentums = weightUpdates

                biasUpdates = self.momentum * layer.biasMomentums - self.currentlearningRate * layer.dbiases
                layer.biasMomentums = biasUpdates
            else:
                weightUpdates = -self.currentlearningRate * layer.dweights
                biasUpdates = -self.currentlearningRate * layer.dbiases

            layer.weights += weightUpdates
            layer.biases += biasUpdates

    def postUpdateParams(self):
        self.iterations += 1
