#!/usr/bin/env python3

from jagerml.helper import *
from jagerml.layers import Input
from jagerml.activations import Softmax, SoftmaxLossCrossentropy
from jagerml.evaluate import LossCategoricalCrossentropy


class Model:

    def __init__(self):
        self.layers = []
        self.softmaxClassifierOutput = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def train(self, X, y, epochs=1, verbose=1, validationData=None, batchSize=None):

        self.accuracy.init(y)
        trainSteps = 1

        if validationData is not None:
            validationSteps = 1
            X_val, y_val = validationData

        if batchSize is not None:
            trainSteps = len(X) // batchSize
            if trainSteps * batchSize < len(X):
                trainSteps += 1

            if validationData is not None:
                validationSteps = len(X_val) // batchSize

            if validationSteps * batchSize < len(X_val):
                validationSteps += 1

        for epoch in range(epochs):
            print("[Epoch] {} \n".format(epoch))
            self.loss.newPass()
            self.accuracy.newPass()

            for step in range(trainSteps):

                if batchSize is None:
                    batchX = X
                    batchy = y
                else:
                    batchX = X[step * batchSize:(step + 1) * batchSize]
                    batchy = y[step * batchSize:(step + 1) * batchSize]

                output = self.forward(batchX, training=True)
                dataLoss, regularizationLoss = self.loss.calculate(output, batchy, useRegularization=True)
                loss = dataLoss + regularizationLoss

                finalPredictions = self.outputLayerActivation.predictions(output)
                finalAccuracy = self.accuracy.calculate(finalPredictions, batchy)

                self.backward(output, batchy)

                self.optimizer.preUpdateParams()
                for layer in self.trainablelayers:
                    self.optimizer.updateParams(layer)
                self.optimizer.postUpdateParams()

                if verbose >= 1:
                    if step % verbose == 0:
                        print("[{:<3}] acc{:.2f} loss {:.2f} dataL {:.2f} lr {:.7f}".format(step,
                                                                                            finalAccuracy,
                                                                                            loss,
                                                                                            dataLoss,
                                                                self.optimizer.currentlearningRate))
            epochDataLoss, epochRegularizationLoss = self.loss.calculateAccumulated(useRegularization=True)
            epochLoss = epochDataLoss + epochRegularizationLoss
            epochAccuracy = self.accuracy.calculateAccumulated()
            print("[Steps] {} :".format(step))
            print("> acc {} loss {} dataLoss {} lr {}".format(finalAccuracy,
                                                              loss,
                                                              dataLoss,
                                                              self.optimizer.currentlearningRate))

            if validationData is not None:
                self.evaluate(*validationData, batchSize=batchSize)

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

            self.layers[-1].dinputs = self.softmaxClassifierOutput.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            self.loss.backward(output, y)

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def evaluate(self, X_val, y_val, batchSize=None):
        validationSteps = 1
        if batchSize is not None:
            validationSteps = len(X_val) // batchSize

            if validationSteps * batchSize < len(X_val):
                validationSteps += 1

            self.loss.newPass()
            self.accuracy.newPass()
        else:
            print("[Evaluate] :")

        for step in range(validationSteps):
            if batchSize is None:
                batchX = X_val
                batchy = y_val
            else:
                batchX = X_val[step * batchSize:(step + 1) * batchSize]
                batchy = y_val[step * batchSize:(step + 1) * batchSize]

            output = self.forward(batchX, training=False)
            self.loss.calculate(output, batchy)

            finalPredictions = self.outputLayerActivation.predictions(output)
            self.accuracy.calculate(finalPredictions, batchy)

        validationLoss = self.loss.calculateAccumulated()
        validationAccuracy = self.accuracy.calculateAccumulated()
        print("* acc {} loss {}".format(validationAccuracy, validationLoss))

    def predict(self, X, batchSize=None):
        predictionSteps = 1

        if batchSize is not None:
            predictionSteps = len(X) // batchSize
            if predictionSteps * batchSize < len(X):
                predictionSteps += 1

        output = []

        for step in range(predictionSteps):
            if batchSize is None:
                batchX = X
            else:
                batchX = X[step * batchSize:(step + 1) * batchSize]

            batchOutput = self.forward(batchX, training=False)

            output.append(batchOutput)
        return np.vstack(output)
