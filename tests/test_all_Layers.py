#!/usr/bin/env python3

import logging
import pytest
# datasets
from sklearn import datasets
from jagerml.layers import Dense

logger = logging.getLogger(__name__)

@pytest.fixture
def iris_data():
    return datasets.load_iris().data

@pytest.mark.parametrize("inputs, neurons", [
    (4, 1),
    (4, 10),
    (4, 100),
    (4, 1000),
])
def test_simple_dense(inputs, neurons, iris_data):
    dense = Dense(inputs, neurons)
    assert dense.forward(iris_data) == True