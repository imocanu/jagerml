#!/usr/bin/env python3

from jagerml.model import Model
from jagerml.layers import Dense, \
    Dropout
from jagerml.activations import ReLU, \
    Softmax, \
    SoftmaxLossCrossentropy, \
    Sigmoid, \
    Linear
from jagerml.evaluate import LossCategoricalCrossentropy, \
    LossBinaryCrossentropy, \
    MeanSquaredError, \
    MeanAbsoluteError, \
    AccuracyRegression, \
    AccuracyCategorical
from jagerml.optimizers import SGD, \
    AdaGrad, \
    RMSprop, \
    Adam
from jagerml.helper import *


def runModel():
    print("[**][run Model - MNIST]")
    from tensorflow import keras
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    keys = np.array(range(X_train.shape[0]))
    np.random.shuffle(keys)
    X_train = X_train[keys]
    y_train = y_train[keys]

    X_train = (X_train.reshape(X_train.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

    model = Model()
    model.add(Dense(X_train.shape[1], 128))
    model.add(ReLU())
    model.add(Dense(128, 128))
    model.add(ReLU())
    model.add(Dense(128, 128))
    model.add(Softmax())

    model.set(
        loss=LossCategoricalCrossentropy(),
        optimizer=Adam(decay=1e-4),
        accuracy=AccuracyCategorical()
    )

    model.fit()
    model.train(X_train, y_train, epochs=1, verbose=1, validationData=(X_test, y_test), batchSize=128)
    model.evaluate(X_test, y_test)

    confidences = model.predict(X_test[:20])
    predictions = model.outputLayerActivation.predictions(confidences)
    print("test :",y_test[:20])
    print("pred :", predictions)


if __name__ == "__main__":
    runModel()
