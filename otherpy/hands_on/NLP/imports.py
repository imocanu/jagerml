import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import seaborn as sns
import os
import matplotlib as mpl
import pandas as pd
import datetime
import re
import string
import tqdm
import io
import PIL
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import collections
import random
import time
import json
from glob import glob
from PIL import Image
import pickle


def versions():
    print("[*] tf   Version :", tf.__version__)
    print("[*] tfds Version :", tfds.__version__)
    assert tf.__version__ >= "2.3.0"
    tf.random.set_seed(356)
    np.random.seed(356)
    sns.set()
    print("[*] tf random is : 356")
    print("[*] np random is : 356")


def check_other_gpu():
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        print("[#] Num of GPU in system : ", len(tf.config.experimental.list_physical_devices('GPU')))
        print("[*] is_gpu_available()   :", tf.test.is_gpu_available())
        config = tf.compat.v1.ConfigProto()
        fraction = 0.9
        config.gpu_options.per_process_gpu_memory_fraction = fraction
        print("  [*] per_process_gpu_memory_fraction is ", fraction)
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
    print("=" * 50)
    versions()
    set_proxy(proxy=proxy)
    check_other_gpu()
    print("=" * 50)


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


def plot_history_with_dataframe(history, epochs):
    print(history.history.keys())
    print(len(history.history))
    print(type(history))
    print(type(history.history))
    stereo = False
    for metric in history.history.keys():
        if "val" in metric:
            stereo = True
        print("> metric :", metric)

    # fig, axs = plt.subplots(len(history.history))
    # for i in range(len(history.history)):
    #     axs[i].plot(range(1, epochs + 1), history.history[metric])
    # plt.show()

    cm = pd.DataFrame(history.history, index=range(1, epochs+1))
    print(cm.head())
    cm.plot(xlabel="Epoch", ylabel="Metric", layout=(1, 2), subplots=True)
    plt.show()


# ==== Callbacks =========
class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nLoss val/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))
        print("Acc val/train: {:.2f}".format(logs["val_sparse_categorical_accuracy"] /
                                             logs["sparse_categorical_accuracy"]))


run_logdir = os.path.join(os.curdir, "logs")
cb_tensorboard = keras.callbacks.TensorBoard(run_logdir)
cb_checkpoint = keras.callbacks.ModelCheckpoint("model_cb.h5",
                                                save_best_only=True)
cb_early_stopping = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True)
cb_print_val_train_ratio = PrintValTrainRatioCallback()

# exponential_decay_fn = exponential_decay(lr0=0.01, s=20)
# lr_scheduler = keras.callbacks.LearningRateScheduler(piecewise_constant_fn)
# lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
# lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
