#!/usr/bin/env python3

import time
import numpy as np
from numpy.testing import assert_almost_equal
from jagerml.tensor import Tensor

np.random.seed(156)


def test_activations():
    x = np.random.randn(5, 5).astype(np.float32)
    y = np.random.randn(5, 5).astype(np.float32)
    xy = np.hstack((x, y))
    yx = np.vstack((x, y))
    n = np.ndarray((2, 2), dtype=np.float32)

    t = Tensor()
    print(t)

    t = Tensor(n)
    print(t)

    t2 = t
    print(t2)
    print(type(t2))
    print(t2.shape)

    t = Tensor.zeros(4)
    print(t)
    print(t.shape)

    t = Tensor.zeros_like(xy)
    print(t)
    print(t.shape)

    t = Tensor.zeros_like((yx, yx))
    print(t)
    print(t.shape)


if __name__ == "__main__":
    test_activations()
