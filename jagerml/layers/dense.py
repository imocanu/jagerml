#!/usr/bin/env python3

from jagerml.imports import *
from jagerml.helper import *

class Dense:
    def __init__(self, nInputs, nNeurons):
        self.weights = np.random.randn(nInputs, nNeurons)
        self.biases  = np.zeros((1, nNeurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

def runTest1():
    print("[**][test][dense_layer]")
    iris = getData()
    data = iris.data
    target = iris.target

    print(iris.keys())
    print(iris.target_names)
    print(iris.feature_names)
    print(iris.data.shape)
    print(iris.target.shape)