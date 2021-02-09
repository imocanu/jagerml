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
import sys
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

RANDOM = SEED = 356
CHECK_DATATSET = 0


def versions():
    print("[*] python Version :", sys.version)
    print("[*] tf   Version   :", tf.__version__)
    print("[*] tfds Version   :", tfds.__version__)
    assert tf.__version__ >= "2.3.0"
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    sns.set()
    # sns.set_style("whitegrid")
    # plt.style.use("fivethirtyeight")
    print("[*] tf random is   :", SEED)
    print("[*] np random is   :", SEED)


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


def set_proxy(proxy=None):
    print("[*] Proxy is       :", proxy)
    if proxy is not None:
        os.environ["http_proxy"] = proxy
        os.environ["HTTP_PROXY"] = proxy
        os.environ["https_proxy"] = proxy
        os.environ["HTTPS_PROXY"] = proxy


def check_version_proxy_gpu(proxy=None):
    print("=" * 60)
    versions()
    set_proxy(proxy=proxy)
    check_other_gpu()
    print("=" * 60)


def check_dataset(dataset):
    global CHECK_DATATSET
    CHECK_DATATSET += 1
    if isinstance(dataset, np.ndarray):
        print("[", CHECK_DATATSET, "] np array :", type(dataset), " - ", len(dataset))
    elif isinstance(dataset, dict):
        print("[", CHECK_DATATSET, "] DICT :", type(dataset), " - ", len(dataset))
        print("[*]      > key :", dataset.keys())
        for any_key in dataset.keys():
            print("<KEY>", any_key, "<TYPE>", type(dataset[any_key]), "<LEN>", len(dataset[any_key]))
            check_dataset(dataset[any_key])
    elif isinstance(dataset, pd.core.series.Series):
        print("[", CHECK_DATATSET, "] SERIES :", type(dataset), " - ", len(dataset))
    elif isinstance(dataset, list):
        print("[", CHECK_DATATSET, "] LIST :", type(dataset), " - ", len(dataset))
    elif isinstance(dataset, tuple):
        print("[", CHECK_DATATSET, "] TUPLE :", type(dataset), " - ", len(dataset))
    elif isinstance(dataset, tf.data.Dataset):
        print("[", CHECK_DATATSET, "] DATASET :", type(dataset))
        try:
            print("       >LEN :", len(dataset))
        except:
            print("     <EXCEPTION_TAKE_1> :", dataset.take(1))
    else:
        print("[", CHECK_DATATSET, "] ELSE :", type(dataset), " - ", len(dataset))


def plot_history_per_keys(history, epochs):
    key_counter = 0
    has_val = False
    metrics = []
    print("[#] PLOT = HISTORY = EPOCHS(", epochs, ")")
    for metric in history.history.keys():
        if "val" in metric:
            has_val = True
        key_counter += 1
        metrics.append(metric)
        print(" {metric} :", metric)

    if key_counter == 1 and has_val is False:
        fig = plt.figure()
        plt.plot(range(1, epochs + 1), history.history[metrics[0]], label='training')
        plt.legend(loc=0)
        plt.xlabel('epochs')
        plt.ylabel(metrics[0])
        plt.xlim([1, epochs])
        plt.grid(True)
        plt.title(metrics[0])

    elif key_counter == 2 and has_val is False:
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        metric = metrics[0]
        plt.plot(range(1, epochs + 1), history.history[metric], label='training')
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.ylim(None, 1)

        plt.subplot(1, 2, 2)
        metric = metrics[1]
        plt.plot(range(1, epochs + 1), history.history[metric], label='training')
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.ylim(0, None)
        plt.grid(True)
        plt.legend(loc=0)
        plt.title(metrics[0] + "-" + metrics[1])

    elif key_counter == 2 and has_val is True:
        fig = plt.figure()
        metric = metrics[0]
        plt.plot(range(1, epochs + 1), history.history[metric], label='training')
        metric = metrics[1]
        plt.plot(range(1, epochs + 1), history.history[metric], label='training')
        plt.legend(loc=0)
        plt.xlabel('epochs')
        plt.ylabel(metrics[0] + metrics[1])
        plt.xlim([1, epochs])
        plt.grid(True)
        plt.title(metrics[0] + "-" + metrics[1])

    elif key_counter > 2 and has_val is True:
        if key_counter == 4:
            plt.figure(figsize=(16, 8))
            plt.subplot(1, 2, 1)
            metric1 = metrics[0]
            metric2 = metrics[1]
            plt.plot(range(1, epochs + 1), history.history[metric1])
            plt.plot(range(1, epochs + 1), history.history[metric2])
            plt.xlabel("Epochs")
            plt.ylabel(metric)
            plt.ylim(None, 1)

            plt.subplot(1, 2, 2)
            metric1 = metrics[2]
            metric2 = metrics[3]
            plt.plot(range(1, epochs + 1), history.history[metric1])
            plt.plot(range(1, epochs + 1), history.history[metric2])
            plt.xlabel("Epochs")
            plt.ylabel(metric)
            plt.ylim(0, None)
            plt.grid(True)
            plt.legend(loc=0)
        else:
            print("[!] NO PLOT for total keys :", key_counter)
    else:
        print("[!] NO PLOT selected !!!")


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

    cm = pd.DataFrame(history.history, index=range(1, epochs + 1))
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
