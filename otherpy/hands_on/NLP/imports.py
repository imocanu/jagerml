import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib as mpl
import pandas as pd
import os
import datetime


def check_version():
    np.random.seed(42)
    tf.random.set_seed(42)
    sns.set()
    print("TF version :", tf.__version__)
    assert tf.__version__ >= "2.0"
    if not tf.config.list_physical_devices('GPU'):
        print("[*] No GPU was detected !!!")