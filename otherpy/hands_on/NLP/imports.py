import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import seaborn as sns
import matplotlib as mpl
import pandas as pd
import os
import datetime
import re
import string
import tqdm
import io


def versions():
    print("[*] tf   Version :", tf.__version__)
    print("[*] tfds Version :", tfds.__version__)
    assert tf.__version__ >= "2.3.0"
    tf.random.set_seed(356)
    np.random.seed(356)
    sns.set()
    print("[*] tf random is : 356")


def check_other_gpu():
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        print("[#] Num of GPU in system : ", len(tf.config.experimental.list_physical_devices('GPU')))
        config = tf.compat.v1.ConfigProto()
        fraction = 0.9
        config.gpu_options.per_process_gpu_memory_fraction = fraction
        print(" [*] per_process_gpu_memory_fraction is ", fraction)
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    else:
        print("[#] NO GPU available !!!!")


def set_proxy(proxy=""):
    print("[*] Proxy is : ", proxy)
    os.environ["http_proxy"] = proxy
    os.environ["HTTP_PROXY"] = proxy
    os.environ["https_proxy"] = proxy
    os.environ["HTTPS_PROXY"] = proxy


def check_version_proxy_gpu(proxy=""):
    print("=" * 40)
    versions()
    set_proxy(proxy=proxy)
    check_other_gpu()
    print("=" * 40)


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


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])


def plot_history(history):
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    plt.ylim(0, None)


def plot_history_without_val(history, epochs):
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    metric = 'accuracy'
    plt.plot(range(1, epochs + 1), history.history[metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.ylim(None, 1)

    plt.subplot(1, 2, 2)
    metric = 'loss'
    plt.plot(range(1, epochs + 1), history.history[metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.ylim(0, None)
    plt.grid(True)
    plt.legend(loc=0)
