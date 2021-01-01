from jagerml import imports
from jagerml.layers import Dense
from jagerml.activations import relu, softmax
from jagerml.evaluate import evaluate

class Evaluate:
    def __init__(self):
        self.LossCategoricalCrossentropy = evaluate.LossCategoricalCrossentropy

class Layers:
    def __init__(self):
        self.Dense = Dense

class Activations:
    def __init__(self):
        self.ReLU = relu.ReLU
        self.Softmax = softmax.Softmax

class ML:
    def __init__(self):
        self.layers = Layers
        self.activations = Activations
        self.evaluate = Evaluate