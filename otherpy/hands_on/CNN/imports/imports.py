import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds


def versions():
    print("[*] tf   Version :", tf.__version__)
    print("[*] tfds Version :", tfds.__version__)
    tf.random.set_seed(356)
    print("[*] tf random is : 356")


def check_other_gpu():
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        print("[#] Num of GPU in system : ", len(tf.config.experimental.list_physical_devices('GPU')))
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    else:
        print("[#] NO GPU available !!!!")


def set_proxy(proxy="http://proxy.ka.intel.com:911"):
    os.environment["http_proxy"] = proxy
    os.environment["HTTP_PROXY"] = proxy
    os.environment["https_proxy"] = proxy
    os.environment["HTTPS_PROXY"] = proxy


def plot_fig(history, epochs):
    fig = plt.figure()
    plt.plot(range(1, epochs + 1), history.history['val_accuracy'], label='validation')
    plt.plot(range(1, epochs + 1), history.history['accuracy'], label='training')
    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.xlim([1, epochs])
    plt.grid(True)
    plt.title("Accuracy")
    plt.show()
