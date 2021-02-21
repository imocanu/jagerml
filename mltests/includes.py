import logging
import pytest
# datasets
from sklearn import datasets
from jagerml.layers import Dense
import time
import sys
import os
import numpy as np

from numpy.testing import assert_almost_equal, assert_equal

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from mltests import TorchDenseLayer

np.random.seed(156)
tf.random.set_seed(156)
torch.random.manual_seed(156)

logger = logging.getLogger(__name__)

RANDOM = SEED = 356
CHECK_DATATSET = 0


def versions():
    print("[*] python Version :", sys.version)
    print("[*] np random is   :", SEED)


def set_proxy(proxy=None):
    print("[*] Proxy is       :", proxy)
    if proxy is not None:
        os.environ["http_proxy"] = proxy
        os.environ["HTTP_PROXY"] = proxy
        os.environ["https_proxy"] = proxy
        os.environ["HTTPS_PROXY"] = proxy


def check_version_proxy(proxy=None):
    print("=" * 60)
    versions()
    set_proxy(proxy=proxy)
    print("=" * 60)


def random_tensor(shape, standardize=False):
    offset = np.random.randint(-300, 300, shape)
    X = np.random.rand(*shape) + offset

    if standardize:
        eps = np.finfo(float).eps
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + eps)
    return X


def stochastic_matrix(features, labels, binary=False):
    sm = np.random.rand(features, labels)
    sm /= sm.sum(axis=1, keepdims=True)

    if binary:
        sm = np.random.randint(2, size=(features, labels))
        print("SM :", sm)

    return sm


def one_hot_matrix(n_examples, n_classes):
    X = np.eye(n_classes)
    X = X[np.random.choice(n_classes, n_examples)]
    return X
