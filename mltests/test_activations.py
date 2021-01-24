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


def torch_gradient_generator(fn, **kwargs):
    def get_grad(z):
        z1 = torch.autograd.Variable(torch.from_numpy(z), requires_grad=True)
        z2 = fn(z1, **kwargs).sum()
        z2.backward()
        grad = z1.grad.numpy()
        return grad

    return get_grad


def random_tensor(shape, standardize=False):
    offset = np.random.randint(-300, 300, shape)
    X = np.random.rand(*shape) + offset

    if standardize:
        eps = np.finfo(float).eps
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + eps)
    return X


def stochastic_matrix(features, labels):
    sm = np.random.rand(features, labels)
    sm /= sm.sum(axis=1, keepdims=True)
    return sm


def test_leaky_relu(loops=100):
    from jagerml.activations import LeakyReLU
    print("[*] LeakyReLU : forward & backward")
    start = time.time()

    for _ in range(loops):
        features = np.random.randint(100, 1000)
        labels = np.random.randint(100, 1000)
        alpha = np.random.uniform(0, 100)
        sm = stochastic_matrix(features, labels)
        rt = random_tensor((features, labels))

        leaky_relu = LeakyReLU(alpha=alpha)
        leaky_relu.forward(sm, None)
        leaky_relu.backward(rt)
        torch_test = torch_nn.leaky_relu(torch.FloatTensor(sm), alpha).numpy()
        tf_test = tf.nn.leaky_relu(tf.convert_to_tensor(sm), alpha).numpy()
        assert_almost_equal(leaky_relu.output, tf_test)
        assert_almost_equal(leaky_relu.output, torch_test)

        torch_test = torch_gradient_generator(torch_nn.leaky_relu,
                                              negative_slope=alpha)
        assert_almost_equal(leaky_relu.dinputs, torch_test(rt), decimal=6)

    end = time.time()
    print("test pass in {:0.2f} s".format(end - start))


def test_leaky_relu_base(loops=100):
    from jagerml.activations import LeakyReLUbase
    print("[*] LeakyReLU : forward & backward")
    start = time.time()

    for _ in range(loops):
        features = np.random.randint(100, 1000)
        labels = np.random.randint(100, 1000)
        alpha = np.random.uniform(0, 100)
        sm = stochastic_matrix(features, labels)
        rt = random_tensor((features, labels))

        leaky_relu = LeakyReLUbase(alpha=alpha)
        torch_test = torch_nn.leaky_relu(torch.FloatTensor(sm), alpha).numpy()
        tf_test = tf.nn.leaky_relu(tf.convert_to_tensor(sm), alpha).numpy()
        assert_almost_equal(leaky_relu.fn(sm), tf_test)
        assert_almost_equal(leaky_relu.fn(sm), torch_test)

        torch_test = torch_gradient_generator(torch_nn.leaky_relu,
                                              negative_slope=alpha)
        assert_almost_equal(leaky_relu.grad(rt), torch_test(rt), decimal=6)

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


def test_relubase(loops=100):
    from jagerml.activations import ReLUbase
    print("[*] ReLU")
    start = time.time()
    for _ in range(loops):
        features = np.random.randint(100, 1000)
        labels = np.random.randint(100, 1000)
        sm = stochastic_matrix(features, labels)

        relu = ReLUbase()
        output = relu.fn(sm)
        torch_test = torch_nn.relu(torch.FloatTensor(sm)).numpy()
        tf_test = tf.nn.relu(tf.convert_to_tensor(sm)).numpy()
        assert_almost_equal(output, tf_test)
        assert_almost_equal(output, torch_test)
    end = time.time()
    print("test pass in {:0.2f} s".format(end - start))


def test_softmaxbase(loops=100):
    from jagerml.activations import Softmaxbase
    print("[*] ReLU")
    start = time.time()
    for _ in range(loops):
        features = np.random.randint(100, 1000)
        labels = np.random.randint(100, 1000)
        sm = stochastic_matrix(features, labels)

        softmax = Softmaxbase()
        output = softmax.fn(sm)
        torch_test = torch_nn.softmax(torch.FloatTensor(sm), dim=1).numpy()
        tf_test = tf.nn.softmax(tf.convert_to_tensor(sm)).numpy()
        assert_almost_equal(output, tf_test)
        assert_almost_equal(output, torch_test)
    end = time.time()
    print("test pass in {:0.2f} s".format(end - start))


def test_activations():
    #test_leaky_relu()
    test_leaky_relu_base()
    #test_relu()
    #test_relubase()
    #test_softmaxbase()


if __name__ == "__main__":
    test_activations()