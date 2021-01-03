#!/usr/bin/env python3

from jagerml.helper import *


class Adam:
    def __init__(self, learningRate=0.001, decay=0., epsilon=1e-7, beta1 = 0.9,beta2= 0.999):
        self.learningRate = learningRate
        self.currentlearningRate = learningRate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def preUpdateParams(self):
        if self.decay:
            self.currentlearningRate = self.learningRate * \
                                       ( 1. / (1. + self.decay * self.iterations))

    def updateParams(self, layer):
        if not hasattr (layer, 'weight_cache'):
            layer.weightMomentums = np.zeros_like(layer.weights)
            layer.weightCache = np.zeros_like(layer.weights)
            layer.biasMomentums = np.zeros_like(layer.biases)
            layer.biasCache = np.zeros_like(layer.biases)

        layer.weightMomentums = self.beta1 * layer.weightMomentums + (1 - self.beta1) * layer.dweights
        layer.biasMomentums = self.beta1 * layer.biasMomentums + (1 - self.beta1) * layer.dbiases

        weightMomentumsCorrected = layer.weightMomentums / (1 - self.beta1 ** (self.iterations + 1))
        biasMomentumsCorrected = layer.biasMomentums / (1 - self.beta1 ** (self.iterations + 1))

        layer.weightCache = self.beta2 * layer.weightCache + (1 - self.beta2) * layer.dweights ** 2
        layer.biasCache = self.beta2 * layer.biasCache + (1 - self.beta2) * layer.dbiases ** 2

        weight_cache_corrected = layer.weightCache / (1 - self.beta2 ** (self.iterations + 1))
        bias_cache_corrected = layer.biasCache / (1 - self.beta2 ** (self.iterations + 1))
        layer.weights += -self.currentlearningRate * \
                          weightMomentumsCorrected / \
                          (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.currentlearningRate * biasMomentumsCorrected  /\
                        (np.sqrt(bias_cache_corrected) + self.epsilon)

    def postUpdateParams(self):
        self.iterations += 1