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
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    else:
        print("[#] NO GPU available !!!!")


def set_proxy(proxy=""):
    print("[*] Proxy is : ", proxy)
    os.environ["http_proxy"] = proxy
    os.environ["HTTP_PROXY"] = proxy
    os.environ["https_proxy"] = proxy
    os.environ["HTTPS_PROXY"] = proxy


def check_version_proxy_gpu():
    versions()
    # set_proxy()
    check_other_gpu()
