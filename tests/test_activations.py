#!/usr/bin/env python3

import time
import numpy as np

from numpy.testing import assert_almost_equal

import tensorflow as tf
import torch
import torch.nn.functional as torch_nn


np.random.seed(156)
tf.random.set_seed(156)
torch.random.manual_seed(156)


def stochastic_matrix(features, labels):
    sm = np.random.rand(features, labels)
    sm /= sm.sum(axis=1, keepdims=True)
    return sm


def test_leaky_relu(loops=100):
    from jagerml.activations import LeakyReLU
    print("[*] LeakyReLU")
    start = time.time()
    for _ in range(loops):
        features = np.random.randint(100, 1000)
        labels = np.random.randint(100, 1000)
        sm = stochastic_matrix(features, labels)
        alpha = np.random.uniform(0, 100)

        leaky_relu = LeakyReLU(alpha=alpha)
        leaky_relu.forward(sm, None)
        torch_test = torch_nn.leaky_relu(torch.FloatTensor(sm), alpha).numpy()
        tf_test = tf.nn.leaky_relu(tf.convert_to_tensor(sm), alpha).numpy()
        assert_almost_equal(leaky_relu.output, tf_test)
        assert_almost_equal(leaky_relu.output, torch_test)
    end = time.time()
    print("test pass in {:0.2f} s".format(end - start))


def test_relu(loops=100):
    from jagerml.activations import ReLU
    print("[*] ReLU")
    start = time.time()
    for _ in range(loops):
        features = np.random.randint(100, 1000)
        labels = np.random.randint(100, 1000)
        sm = stochastic_matrix(features, labels)

        relu = ReLU()
        relu.forward(sm, None)
        torch_test = torch_nn.relu(torch.FloatTensor(sm)).numpy()
        tf_test = tf.nn.relu(tf.convert_to_tensor(sm)).numpy()
        assert_almost_equal(relu.output, tf_test)
        assert_almost_equal(relu.output, torch_test)
    end = time.time()
    print("test pass in {:0.2f} s".format(end - start))


def test_activations():
    test_leaky_relu()
    test_relu()


if __name__ == "__main__":
    test_activations()
