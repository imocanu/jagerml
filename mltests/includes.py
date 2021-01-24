import logging
import pytest
# datasets
from sklearn import datasets
from jagerml.layers import Dense
import time
import numpy as np

from numpy.testing import assert_almost_equal

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from mltests import TorchDenseLayer

np.random.seed(156)
tf.random.set_seed(156)
torch.random.manual_seed(156)

logger = logging.getLogger(__name__)