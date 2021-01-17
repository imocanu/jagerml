#!/usr/bin/env python3

import sys
import inspect
import functools
import os
import numpy as np
from time import time
from collections import defaultdict, OrderedDict


class Tensor:
    training = True
    ops = defaultdict(dict)

    def __init__(self, data=None):
        self.data = self._parse_data(data)

    def __repr__(self):
        print("__reps__")
        return self.data

    def __str__(self):
        return f"<Tensor {self.data!r} >"

    @property
    def assign(self, tensor):
        self.data = tensor.data

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @staticmethod
    def _parse_data(data):
        if data is None:
            data = np.empty((0,0), dtype=np.float32)
            return data

        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        if data.dtype != np.float32:
            print("[!!!] data is NOT float32")

        return data

    @classmethod
    def zeros(cls, *shape):
        return cls(np.zeros(shape, dtype=np.float32))

    @classmethod
    def zeros_like(cls, *shape):
        return cls(np.zeros_like(shape, dtype=np.float32))

    def walk(self, visited: set, nodes: list):
        visited.add(self)
        nodes.append(self)
        return nodes

    def backward(self):
        assert self.shape == (1,)



