#!/usr/bin/env python3

from jagerml.helper import *


class BaseActivation(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, z):
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.fn(z)

    @abstractmethod
    def fn(self, z):
        raise NotImplementedError

    @abstractmethod
    def grad(self, x, **kwargs):
        raise NotImplementedError


class Linearbase(BaseActivation):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.dinputs = None
        super().__init__()

    def __str__(self):
        return "Linearbase"

    def fn(self, z):
        return z

    def grad(self, x):
        return np.ones_like(x)


class Affinebase(BaseActivation):
    def __init__(self, slope=1, intercept=0):
        self.slope = slope
        self.intercept = intercept
        super().__init__()

    def __str__(self):
        return "Affinebase"

    def fn(self, z):
        return self.slope * z + self.intercept

    def grad(self, x):
        return self.slope * np.ones_like(x)


class ReLUbase(BaseActivation):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ReLUbase"

    def fn(self, z):
        return np.clip(z, 0, np.inf)

    def grad(self, x):
        return (x > 0).astype(int)

    def grad2(self, x):
        return np.zeros_like(x)


class LeakyReLUbase(BaseActivation):
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.output = None
        self.dinputs = None
        super().__init__()

    def __str__(self):
        return "LeakyReLUbase"

    def fn(self, z):
        self.output = z.copy()
        self.output[z < 0] = self.output[z < 0] * self.alpha
        return self.output

    def grad(self, x):
        self.dinputs = np.ones_like(x)
        self.dinputs[x < 0] *= self.alpha
        return self.dinputs

    def grad2(self, x):
        return np.zeros_like(x)


class Softmaxbase(BaseActivation):
    def __init__(self):
        self.output = None
        self.dinputs = None
        super().__init__()

    def __str__(self):
        return "Softmaxbase"

    def fn(self, z):
        self.inputs = z
        expVals = np.exp(z - np.max(z,
                                    axis=1,
                                    keepdims=True))
        # normalize
        normalize = expVals / np.sum(expVals,
                                     axis=1,
                                     keepdims=True)
        self.output = normalize
        return self.output

    def grad(self, x):
        self.dinputs = np.empty_like(x)

        for index, (singleOutput, singleDvalues) in enumerate(zip(self.output, x)):
            singleOutput = singleOutput.reshape(-1, 1)
            jacobianMatrix = np.diagflat(singleOutput) - np.dot(singleOutput, singleOutput.T)
            self.dinputs[index] = np.dot(jacobianMatrix, singleDvalues)
        return self.dinputs

    def grad2(self, x):
        return np.zeros_like(x)
