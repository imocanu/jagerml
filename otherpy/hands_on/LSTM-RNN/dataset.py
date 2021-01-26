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
        print("[*] No GPU was detected. LSTMs and CNNs can be very slow without a GPU.")


def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)  # + noise
    return series[..., np.newaxis].astype(np.float32)


def dataset_keras_utils_get_file_path(url="", fname=""):
    zip_path = tf.keras.utils.get_file(origin=url, fname=fname, extract=True)
    return zip_path
