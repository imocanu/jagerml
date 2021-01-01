#!/usr/bin/env python3

from jagerml.imports import *
from jagerml.helper import *

class Loss:

    def calculate(self, output, y):
        losses = self.forward(output, y)
        meanLoss = np.mean(losses)
        return meanLoss

class LossCategoricalCrossentropy(Loss):

    def forward(self, yPred, yTrue):
        samples = len(yPred)
        yPredClipp = np.clip(yPred, 1e-7, 1 - 1e-7)

        if len(yTrue.shape) == 1:
            predicted = yPredClipp[range[samples], yTrue]
        elif len(yTrue.shape) == 2:
            predicted = np.sum(yPredClipp * yTrue, axis=1)

        loss = -np.log(predicted)
        return loss

