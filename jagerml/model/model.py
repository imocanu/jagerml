#!/usr/bin/env python3

from jagerml.helper import *
from jagerml.layers import Input
from jagerml.activations import Softmax, SoftmaxLossCrossentropy
from jagerml.evaluate import LossCategoricalCrossentropy


class Model:

    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.inputLayer = None
        self.accuracy = None
        self.softmaxClassifierOutput = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def train(self, X, y, epochs=1, verbose=1, validationData=None):

        self.accuracy.init(y)

        for epoch in range(1, epochs+1):
            output = self.forward(X, training=True)

            dataLoss, regularizationLoss = self.loss.calculate(output, y, useRegularization=True)
            loss = dataLoss + regularizationLoss

            predictions = self.outputLayerActivation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            self.backward(output, y)

            self.optimizer.preUpdateParams()
            for layer in self.trainablelayers:
                self.optimizer.updateParams(layer)
            self.optimizer.postUpdateParams()

            if not epoch % verbose:
                print("Epoch {} acc {} loss {} ls {}".format(epoch,
                                                             accuracy,
                                                             loss,
                                                             self.optimizer.currentlearningRate))
        if validationData is not None:
            X_test, y_test = validationData
            output = self.forward(X_test)
            loss = self.loss.calculate(output, y_test)

            predictions = self.outputLayerActivation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_test)

            print("Validation acc {} loss {}", accuracy, loss)


    def fit(self):
        self.inputLayer = Input()
        layerCount = len(self.layers)
        self.trainablelayers = []

        for i in range(layerCount):
            if i == 0:
                self.layers[i].prev = self.inputLayer
                self.layers[i].next = self.layers[i + 1]
            elif i < layerCount - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.outputLayerActivation = self.layers[i]

            if hasattr(self.layers[i], "weights"):
                self.trainablelayers.append(self.layers[i])

            self.loss.rememberTrainableLayers(self.trainablelayers)

        if isinstance(self.layers[-1], Softmax) and \
                isinstance(self.loss, LossCategoricalCrossentropy):
            self.softmaxClassifierOutput = SoftmaxLossCrossentropy()

    def forward(self, X, training):
        self.inputLayer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output

    def backward(self, output, y):

        if self.softmaxClassifierOutput is not None:
            self.softmaxClassifierOutput.backward(output, y)

            self.layers[-1].dinputs = self.softmaxClassifierOutput

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            self.loss.backward(output, y)

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)