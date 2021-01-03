#!/usr/bin/env python3

from jagerml.model import Model
from jagerml.layers import Dense, Dropout
from jagerml.activations import ReLU, Softmax, SoftmaxLossCrossentropy, Sigmoid, Linear
from jagerml.evaluate import LossCategoricalCrossentropy, \
    BinaryCrossentropy, \
    MeanSquaredError, \
    MeanAbsoluteError, \
    AccuracyRegression
from jagerml.optimizers import SGD, AdaGrad, RMSprop, Adam
from jagerml.helper import *


def runModel():
    print("[**][run model]")
    iris = getData()
    data = iris.data
    target = iris.target

    print(iris.keys())
    print(iris.target_names)
    print(iris.feature_names)
    print(iris.data.shape)
    print(iris.target.shape)

    relu = ReLU()
    softmax = Softmax()
    dense1 = Dense(4,8)
    dense2 = Dense(8,3)
    loss = LossCategoricalCrossentropy()

    dense1.forward(data)
    relu.forward(dense1.output)

    dense2.forward(relu.output)
    softmax.forward(dense2.output)

    lossVal = loss.calculate(softmax.output, target)

    print("Input :\n", data[:3])
    print("Softmax :\n", softmax.output[:3])
    print("Loss :", lossVal)

def runModel2():
    print("[**][run model]")
    iris = getData()
    data = iris.data
    target = iris.target

    relu = ReLU()
    softmax = Softmax()
    dense1 = Dense(4,6)
    dense2 = Dense(6,3)
    loss = LossCategoricalCrossentropy()

    maxLoss = 1000
    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases  = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases  = dense2.biases.copy()

    for step in range(10000):
        dense1.weights += 0.05 * np.random.randn(4, 6)
        dense1.biases  += 0.05 * np.random.randn(1, 6)
        dense2.weights += 0.05 * np.random.randn(6, 3)
        dense2.biases  += 0.05 * np.random.randn(1, 3)

        dense1.forward(data)
        relu.forward(dense1.output)
        dense2.forward(relu.output)
        softmax.forward(dense2.output)

        lossVal = loss.calculate(softmax.output, target)

        predictions = np.argmax(softmax.output, axis = 1)
        accuracy = np.mean(predictions==target)

        if lossVal < maxLoss:
            print("New weights for step {} loss {} acc {}".format(step, lossVal, accuracy))
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
            maxLoss = lossVal
        else:
            dense1.weights = dense1.weights.copy()
            dense1.biases = dense1.biases.copy()
            dense2.weights = dense2.weights.copy()
            dense2.biases = dense2.biases.copy()


def runModel3():
    print("[**][run model]")
    iris = getData()
    data = iris.data
    target = iris.target

    dense1 = Dense(4,6)
    relu = ReLU()
    dense2 = Dense(6,3)
    softmaxLoss = SoftmaxLossCrossentropy()
    loss = LossCategoricalCrossentropy()

    sgd = SGD(decay=1e-3, momentum=0.9)
    lossTotal = []
    accTotal  = []

    for epoch in range(10001):
        dense1.forward(data)
        relu.forward(dense1.output)
        dense2.forward(relu.output)
        lossVal = softmaxLoss.forward(dense2.output, target)
        lossTotal.append(lossVal)

        predictions = np.argmax(softmaxLoss.output, axis=1)
        accuracy = np.mean(predictions==target)
        accTotal.append(accuracy)

        if not epoch%100:
            print("Epoch {} loss {} acc {} lr {}".format(epoch,
                                                         lossVal,
                                                         accuracy,
                                                         sgd.currentlearningRate))

        softmaxLoss.backward(softmaxLoss.output, target)
        dense2.backward(softmaxLoss.dinputs)
        relu.backward(dense2.dinputs)
        dense1.backward(relu.dinputs)

        sgd.preUpdateParams()
        sgd.updateParams(dense1)
        sgd.updateParams(dense2)
        sgd.postUpdateParams()

    plt.figure(1)
    plt.plot(np.asarray(range(len(lossTotal))), np.asarray(lossTotal))
    plt.figure(2)
    plt.plot(np.asarray(range(len(accTotal))),  np.asarray(accTotal))
    plt.show()


def runModel4():
    print("[**][run model - AdaGrad]")
    iris = getData()
    data = iris.data
    target = iris.target

    dense1 = Dense(4,6)
    relu = ReLU()
    dense2 = Dense(6,3)
    softmaxLoss = SoftmaxLossCrossentropy()
    loss = LossCategoricalCrossentropy()

    optimizer = AdaGrad(decay=1e-4)
    lossTotal = []
    accTotal  = []
    lrTotal = []

    for epoch in range(10001):
        dense1.forward(data)
        relu.forward(dense1.output)
        dense2.forward(relu.output)
        lossVal = softmaxLoss.forward(dense2.output, target)
        lossTotal.append(lossVal)

        predictions = np.argmax(softmaxLoss.output, axis=1)
        accuracy = np.mean(predictions==target)
        accTotal.append(accuracy)

        if not epoch%1000:
            print("Epoch {} loss {} acc {} lr {}".format(epoch,
                                                         lossVal,
                                                         accuracy,
                                                         optimizer.currentlearningRate))
        lrTotal.append(optimizer.currentlearningRate)

        softmaxLoss.backward(softmaxLoss.output, target)
        dense2.backward(softmaxLoss.dinputs)
        relu.backward(dense2.dinputs)
        dense1.backward(relu.dinputs)

        optimizer.preUpdateParams()
        optimizer.updateParams(dense1)
        optimizer.updateParams(dense2)
        optimizer.postUpdateParams()

    plt.figure(1)
    plt.plot(np.asarray(range(len(lossTotal))), np.asarray(lossTotal))
    plt.figure(2)
    plt.plot(np.asarray(range(len(accTotal))),  np.asarray(accTotal))
    plt.figure(3)
    plt.plot(np.asarray(range(len(lrTotal))),  np.asarray(lrTotal))
    plt.show()


def runModel5():
    print("[**][run model - RMSprop]")
    iris = getData()
    data = iris.data
    target = iris.target

    dense1 = Dense(4,6)
    relu = ReLU()
    dense2 = Dense(6,3)
    softmaxLoss = SoftmaxLossCrossentropy()
    loss = LossCategoricalCrossentropy()

    optimizer = RMSprop(decay=1e-4)
    lossTotal = []
    accTotal  = []
    lrTotal = []

    for epoch in range(10001):
        dense1.forward(data)
        relu.forward(dense1.output)
        dense2.forward(relu.output)
        lossVal = softmaxLoss.forward(dense2.output, target)
        lossTotal.append(lossVal)

        predictions = np.argmax(softmaxLoss.output, axis=1)
        accuracy = np.mean(predictions==target)
        accTotal.append(accuracy)

        if not epoch%1000:
            print("Epoch {} loss {} acc {} lr {}".format(epoch,
                                                         lossVal,
                                                         accuracy,
                                                         optimizer.currentlearningRate))
        lrTotal.append(optimizer.currentlearningRate)

        softmaxLoss.backward(softmaxLoss.output, target)
        dense2.backward(softmaxLoss.dinputs)
        relu.backward(dense2.dinputs)
        dense1.backward(relu.dinputs)

        optimizer.preUpdateParams()
        optimizer.updateParams(dense1)
        optimizer.updateParams(dense2)
        optimizer.postUpdateParams()

    plt.figure(1)
    plt.plot(np.asarray(range(len(lossTotal))), np.asarray(lossTotal))
    plt.figure(2)
    plt.plot(np.asarray(range(len(accTotal))),  np.asarray(accTotal))
    plt.figure(3)
    plt.plot(np.asarray(range(len(lrTotal))),  np.asarray(lrTotal))
    plt.show()


def runModel6():
    print("[**][run model - Adam]")
    iris = getData()
    data = iris.data
    target = iris.target

    dense1 = Dense(4,64)
    relu = ReLU()
    dense2 = Dense(64,3)
    softmaxLoss = SoftmaxLossCrossentropy()
    loss = LossCategoricalCrossentropy()

    optimizer = Adam(learningRate=0.05, decay=5e-7)
    lossTotal = []
    accTotal  = []
    lrTotal = []

    for epoch in range(100):
        dense1.forward(data)
        relu.forward(dense1.output)
        dense2.forward(relu.output)
        lossVal = softmaxLoss.forward(dense2.output, target)
        lossTotal.append(lossVal)

        predictions = np.argmax(softmaxLoss.output, axis=1)
        accuracy = np.mean(predictions==target)
        accTotal.append(accuracy)

        if not epoch%10:
            print("Epoch {} loss {} acc {} lr {}".format(epoch,
                                                         lossVal,
                                                         accuracy,
                                                         optimizer.currentlearningRate))
        lrTotal.append(optimizer.currentlearningRate)
        if accuracy == 1.0:
            print("Epoch {} loss {} acc {} lr {}".format(epoch,
                                                        lossVal,
                                                        accuracy,
                                                        optimizer.currentlearningRate))
            break

        softmaxLoss.backward(softmaxLoss.output, target)
        dense2.backward(softmaxLoss.dinputs)
        relu.backward(dense2.dinputs)
        dense1.backward(relu.dinputs)

        optimizer.preUpdateParams()
        optimizer.updateParams(dense1)
        optimizer.updateParams(dense2)
        optimizer.postUpdateParams()

    plt.figure(1)
    plt.plot(np.asarray(range(len(lossTotal))), np.asarray(lossTotal))
    plt.figure(2)
    plt.plot(np.asarray(range(len(accTotal))),  np.asarray(accTotal))
    plt.figure(3)
    plt.plot(np.asarray(range(len(lrTotal))),  np.asarray(lrTotal))
    plt.show()

def runModel7():
    print("[**][run model - Adam]")
    iris = getData()
    data = iris.data
    target = iris.target

    dense1 = Dense(4, 512, weightL2=5e-4, biasL2=5e-4)
    relu = ReLU()
    dropout = Dropout(0.1)
    dense2 = Dense(512, 3)
    lossActivation = SoftmaxLossCrossentropy()

    optimizer = Adam(learningRate=0.02, decay=5e-7)

    lossTotal = []
    accTotal  = []
    lrTotal = []

    for epoch in range(1000):
        dense1.forward(data)
        relu.forward(dense1.output)
        dropout.forward(relu.output)
        dense2.forward(dropout.output)
        lossVal = lossActivation.forward(dense2.output, target)
        lossTotal.append(lossVal)

        regularizationLoss = lossActivation.loss.regularization(dense1) + \
                             lossActivation.loss.regularization(dense2)

        loss = lossVal + regularizationLoss

        predictions = np.argmax(lossActivation.output, axis=1)
        accuracy = np.mean(predictions==target)
        accTotal.append(accuracy)

        if not epoch%100:
            print("Epoch {} loss {} acc {} lr {} regLoss {}".format(epoch,
                                                                    lossVal,
                                                                    accuracy,
                                                                    optimizer.currentlearningRate,
                                                                    regularizationLoss))
        lrTotal.append(optimizer.currentlearningRate)

        lossActivation.backward(lossActivation.output, target)
        dense2.backward(lossActivation.dinputs)
        dropout.backward(dense2.inputs)
        relu.backward(dense2.dinputs)
        dense1.backward(relu.dinputs)

        optimizer.preUpdateParams()
        optimizer.updateParams(dense1)
        optimizer.updateParams(dense2)
        optimizer.postUpdateParams()

    plt.figure(1)
    plt.plot(np.asarray(range(len(lossTotal))), np.asarray(lossTotal))
    plt.figure(2)
    plt.plot(np.asarray(range(len(accTotal))),  np.asarray(accTotal))
    plt.figure(3)
    plt.plot(np.asarray(range(len(lrTotal))),  np.asarray(lrTotal))
    plt.show()

def runModel8():
    print("[**][run model - Adam]")
    iris = getData()
    data = iris.data
    target = iris.target

    target = target.reshape(-1, 1)
    #print(target)

    dense1 = Dense(4, 64, weightL2=5e-4, biasL2=5e-4)
    relu = ReLU()
    dense2 = Dense(64, 1)
    sigmoid = Sigmoid()

    lossFunction = BinaryCrossentropy()

    optimizer = Adam(learningRate=0.001, decay=5e-7)

    lossTotal = []
    accTotal  = []
    lrTotal = []

    for epoch in range(10000):
        dense1.forward(data)
        relu.forward(dense1.output)
        dense2.forward(relu.output)
        sigmoid.forward(dense2.output)
        lossVal = lossFunction.calculate(sigmoid.output, target)
        lossTotal.append(lossVal)

        regularizationLoss = lossFunction.regularization(dense1) + \
                             lossFunction.regularization(dense2)

        loss = lossVal + regularizationLoss

        predictions = (sigmoid.output > 0.5 ) * 1
        accuracy = np.mean(predictions==target)
        accTotal.append(accuracy)

        if not epoch%100:
            print("Epoch {} loss {} acc {} lr {} regLoss {}".format(epoch,
                                                                    lossVal,
                                                                    accuracy,
                                                                    optimizer.currentlearningRate,
                                                                    regularizationLoss))
        lrTotal.append(optimizer.currentlearningRate)

        lossFunction.backward(sigmoid.output, target)
        sigmoid.backward(lossFunction.dinputs)
        dense2.backward(sigmoid.dinputs)
        relu.backward(dense2.dinputs)
        dense1.backward(relu.dinputs)

        optimizer.preUpdateParams()
        optimizer.updateParams(dense1)
        optimizer.updateParams(dense2)
        optimizer.postUpdateParams()

    plt.figure(1)
    plt.plot(np.asarray(range(len(lossTotal))), np.asarray(lossTotal))
    plt.figure(2)
    plt.plot(np.asarray(range(len(accTotal))),  np.asarray(accTotal))
    plt.figure(3)
    plt.plot(np.asarray(range(len(lrTotal))),  np.asarray(lrTotal))
    plt.show()

def runModel9():
    print("[**][run model - Regression]")
    iris = getData()
    data = iris.data
    target = iris.target

    target = target.reshape(-1, 1)
    #print(target)

    dense1 = Dense(4, 64, weightL2=5e-4, biasL2=5e-4)
    relu = ReLU()
    dense2 = Dense(64, 1)
    linear = Linear()

    lossFunction = MeanSquaredError()
    optimizer = Adam(learningRate=0.01, decay=1e-3)

    lossTotal = []
    accTotal  = []
    lrTotal = []

    accuracyPrecision = np.std(target) / 250

    for epoch in range(10000):
        dense1.forward(data)
        relu.forward(dense1.output)
        dense2.forward(relu.output)
        linear.forward(dense2.output)
        lossVal = lossFunction.calculate(linear.output, target)

        regularizationLoss = lossFunction.regularization(dense1) + \
                             lossFunction.regularization(dense2)

        loss = lossVal + regularizationLoss
        lossTotal.append(loss)

        predictions = linear.output
        accuracy = np.mean(np.absolute(predictions - target) < accuracyPrecision)
        accTotal.append(accuracy)

        if not epoch%100:
            print("Epoch {} loss {} acc {} lr {} regLoss {}".format(epoch,
                                                                    loss,
                                                                    accuracy,
                                                                    optimizer.currentlearningRate,
                                                                    regularizationLoss))
        lrTotal.append(optimizer.currentlearningRate)

        lossFunction.backward(linear.output, target)
        linear.backward(lossFunction.dinputs)
        dense2.backward(linear.dinputs)
        relu.backward(dense2.dinputs)
        dense1.backward(relu.dinputs)

        optimizer.preUpdateParams()
        optimizer.updateParams(dense1)
        optimizer.updateParams(dense2)
        optimizer.postUpdateParams()

    plt.figure(1)
    plt.plot(np.asarray(range(len(lossTotal))), np.asarray(lossTotal))
    plt.figure(2)
    plt.plot(np.asarray(range(len(accTotal))),  np.asarray(accTotal))
    plt.figure(3)
    plt.plot(np.asarray(range(len(lrTotal))),  np.asarray(lrTotal))
    plt.figure(4)
    plt.plot(data[:,2], target)
    plt.plot(data[:, 2], linear.output)
    plt.show()

def runModel10():
    print("[**][run Model + Regression]")
    iris = getData()
    data = iris.data
    target = iris.target

    target = target.reshape(-1, 1)

    model = Model()
    model.add(Dense(4, 64, weightL2=5e-4, biasL2=5e-4))
    model.add(ReLU())
    model.add(Dense(64, 64))
    model.add(ReLU())
    model.add(Dense(64, 1))
    model.add(Linear())

    model.set(
        loss=MeanSquaredError(),
        optimizer=Adam(learningRate=0.005, decay=1e-3),
        accuracy=AccuracyRegression()
    )

    model.fit()
    model.train(data, target, epochs=10000, verbose=100)

if __name__ == "__main__":
    #runModel()
    #runModel2()
    #runModel3()
    #runModel4()
    #runModel5()
    #runModel6()
    #runModel7()
    #runModel8()
    #runModel9()
    runModel10()