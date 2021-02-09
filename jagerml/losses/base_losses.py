#!/usr/bin/env python3

from jagerml.helper import *


class BaseLoss(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss(self, y_true, y_pre):
        pass

    @abstractmethod
    def grad(self, y_true, y_pred, **kwargs):
        pass


class SquaredLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "SquaredError"

    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)

    @staticmethod
    def loss(y, y_pred):
        # return 0.5 * np.linalg.norm(y_pred - y) ** 2
        return np.mean((y - y_pred)**2)

    @staticmethod
    def grad(y, y_pred, z, act_fn):
        return (y_pred - y) * act_fn.grad(z)


class MeanSquaredLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "SquaredError"

    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)

    @staticmethod
    def loss(y, y_pred):
        return np.mean((y - y_pred)**2)

    @staticmethod
    def grad(y, y_pred, z, act_fn):
        return (y_pred - y) * act_fn.grad(z)


class CrossEntropy(BaseLoss):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "CrossEntropy"

    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)

    @staticmethod
    def loss(y, y_pred):
        is_binary(y)
        is_stochastic(y_pred)
        eps = np.finfo(float).eps
        corss_entropy = -np.sum(y * np.log(y_pred + eps))
        return corss_entropy

    @staticmethod
    def grad(y, y_pred):
        is_binary(y)
        is_stochastic(y_pred)
        grad = y_pred - y
        return grad


class BinaryCrossEntropy(BaseLoss):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "BinaryCrossEntropy"

    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)

    @staticmethod
    def loss(y, y_pred):
        is_binary(y)
        is_stochastic(y_pred)
        eps = np.eps
        corss_entropy = -np.sum(y * np.log(y_pred + eps))
        return corss_entropy

    @staticmethod
    def grad(y, y_pred):
        is_binary(y)
        is_stochastic(y_pred)
        grad = y_pred - y
        return grad
