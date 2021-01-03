#!/usr/bin/env python3

from jagerml.helper import *


class Loss:

    def rememberTrainableLayers(self, trainLayers):
        self.trainLayers = trainLayers

    def regularization(self, layer):
        regularizationLoss = 0

        if layer.weightL1 > 0:
            regularizationLoss += layer.weightL1 * np.sum(np.abs(layer.weights))

        if layer.weightL2 > 0:
            regularizationLoss += layer.weightL2 * np.sum(layer.weights * layer.weights)

        if layer.biasL1 > 0:
            regularizationLoss += layer.biasL1 * np.sum(np.abs(layer.biases))

        if layer.biasL2 > 0:
            regularizationLoss += layer.biasL2 * np.sum(layer.biases * layer.biases)

        return regularizationLoss

    def calculate(self, output, y, *,useRegularization=False):
        losses = self.forward(output, y)
        meanLoss = np.mean(losses)
        if not useRegularization:
            return meanLoss

        return meanLoss, self.regularization()


class LossCategoricalCrossentropy(Loss):

    def forward(self, yPred, yTrue):
        samples = len(yPred)
        yPredClipp = np.clip(yPred, 1e-7, 1 - 1e-7)

        if len(yTrue.shape) == 1:
            predicted = yPredClipp[range(samples), yTrue]
        elif len(yTrue.shape) == 2:
            predicted = np.sum(yPredClipp * yTrue, axis=1)

        loss = -np.log(predicted)
        return loss

    def backward(self, dvalues, yTrue):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(yTrue.shape) == 1:
            yTrue = np.eye(labels)[yTrue]

        self.dinputs = -yTrue / dvalues
        self.dinputs = self.dinputs / samples


class BinaryCrossentropy(Loss):

    def forward(self, yPred, yTrue):
        yPredClip = np.clip(yPred, 1e-7, 1 - 1e-7)
        losses = -(yTrue * np.log(yPredClip) + (1 - yTrue) * np.log(1 - yPredClip))
        losses = np.mean(losses, axis=1)

        return losses

    def backward(self, dvalues, yTrue):
        samples = len(dvalues)
        labels = len(dvalues[0])

        dValuesClip = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dinputs = -(yTrue / dValuesClip - (1 - yTrue) / (1 - dValuesClip)) / labels
        self.dinputs = self.dinputs / samples


class MeanSquaredError(Loss):

    def forward(self, yPred, yTrue):
        losses = np.mean((yPred - yTrue) ** 2, axis=-1)
        return losses

    def backward(self, dvalues, yTrue):
        samples = len(dvalues)
        labels = len(dvalues[0])

        self.dinputs = -2 * (yTrue - dvalues) / labels
        self.dinputs = self.dinputs / samples


class MeanAbsoluteError(Loss):

    def forward(selfself, yPred, yTrue):
        losses = np.mean(np.abs(yTrue - yPred), axis=-1)
        return losses

    def backward(self, dvalues, yTrue):
        samples = len(dvalues)
        labels = len(dvalues[0])

        self.dinputs = np.sign(yTrue - dvalues) / labels
        self.dinputs = self.dinputs / samples


class Accuracy:

    def calculate(self, yPred, yTrue):
        comparisons = self.compare(yPred, yTrue)
        acc = np.mean(comparisons)
        return acc


class AccuracyRegression(Accuracy):

    def __init__(self):
        self.precision = None

    def compare(self, yPred, yTrue):
        return np.absolute(yPred - yTrue) < self.precision
