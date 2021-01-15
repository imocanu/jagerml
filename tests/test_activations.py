#!/usr/bin/env python3

import logging
import pytest
# datasets
from sklearn import datasets
from jagerml.layers import Dense

import time
import numpy as np

from numpy.testing import assert_almost_equal
from scipy.special import expit

import torch
import torch.nn.functional as functional


def stochastic_matrix(features, labels):
    X = np.random.rand(features, labels)
    # X /= X.sum(axis=1, keepdims=True)
    return X


def test_leaky_relu(N=50):
    from jagerml.activations import LeakyReLU
    N = 100 if N is None else N
    for _ in range(N):
        n_dims = np.random.randint(1, 500)
        z = stochastic_matrix(1, n_dims)
        alpha = np.random.uniform(0, 100)

        leaky_relu = LeakyReLU(alpha=alpha)
        leaky_relu.forward(z, True)
        torch_test = functional.leaky_relu(torch.FloatTensor(z), alpha).numpy()
        assert_almost_equal(leaky_relu.output, torch_test)
    print("test pass")


def test_relu(N=50):
    from jagerml.activations import ReLU
    N = 100 if N is None else N
    for _ in range(N):
        n_dims = np.random.randint(1, 500)
        z = stochastic_matrix(1, n_dims)

        relu = ReLU()
        relu.forward(z, True)
        torch_test = functional.relu(torch.FloatTensor(z)).numpy()
        assert_almost_equal(relu.output, torch_test)
    print("test pass")


def test_activations():
    print("LeakyReLU")
    test_leaky_relu()
    print("ReLU")
    test_relu()


if __name__ == "__main__":
    test_activations()
