#!/usr/bin/env python3

import jagerml
from helper import parseParams, checkTestID
from jagerml.helper import *

args = parseParams()
ml = jagerml.ML()
layers = ml.layers()
activations = ml.activations()
evaluation = ml.evaluate()

def runTest1():
    print("[**][test{}][model]".format(args.test))
    iris = getData()
    data = iris.data
    target = iris.target

    print(iris.keys())
    print(iris.target_names)
    print(iris.feature_names)
    print(iris.data.shape)
    print(iris.target.shape)

    softmax = activations.Softmax()
    dense = layers.Dense(4,3)

    dense.forward((data))
    softmax.forward(dense.output)

    print("Input :", data[:3])
    print("Dense :", dense.output[:3])
    print("Softmax :", softmax.output[:3])

def runTest2():
    print("[**][test{}][model]".format(args.test))
    iris = getData()
    data = iris.data
    target = iris.target

    print(iris.keys())
    print(iris.target_names)
    print(iris.feature_names)
    print(iris.data.shape)
    print(iris.target.shape)

    softmax = activations.Softmax()
    dense1 = layers.Dense(4,3)
    softmax1 = activations.Softmax
    dense2 = layers.Dense(4,3)
    loss = evaluation.LossCategoricalCrossentropy()

    dense.forward((data))
    softmax.forward(dense.output)

    print("Input :", data[:3])
    print("Dense :", dense.output[:3])
    print("Softmax :", softmax.output[:3])

if(args.test > 0):
    if args.test == 1:
        runTest1()
    if args.test == 2:
        runTest2()
    else:
        print("[!] Test number {} was NOT found !!!".format(args.test))

else:
    print("[*] Run all - NOT implemented")

