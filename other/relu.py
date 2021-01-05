#!/usr/bin/env python3

from jagerml.helper import *

import time
import numpy as np

def index_relu(m):
    m[m < 0] = 0

relus = {
    "max": lambda x: np.maximum(x, 0),
    "in-place max": lambda x: np.maximum(x, 0, x),
    "mul": lambda x: x * (x > 0),
    "abs": lambda x: (abs(x) + x) / 2,
    "fancy index": index_relu,
}

for name, relu in relus.items():
    n_iter = 20
    x = np.random.random((n_iter, 5000, 5000)) - 0.5

    t1 = time.time()
    for i in range(n_iter):
        relu(x[i])
    t2 = time.time()

    print("{:>12s}  {:3.0f} ms".format(name, (t2 - t1) / n_iter * 1000))