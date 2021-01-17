"""

"""

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print(tf.__version__)
tf.random.set_seed(42)
np.random.seed(42)

t = tf.constant([[1.,2.,3.], [4., 5., 6.]])
print(t)
print(t.shape)
print(t + 10)