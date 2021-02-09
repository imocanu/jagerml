#!/usr/bin/env python3

from .imports import *

RANDOM = SEED = 356
CHECK_DATATSET = 0

def getData():
    iris = datasets.load_iris()
    return iris


def check_other_gpu():
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        print("[#] Num of GPU in system : ", len(tf.config.experimental.list_physical_devices('GPU')))
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    else:
        print("[#] NO GPU available !!!!")


def is_binary(x):
    msg = "Matrix must be binary"
    assert np.array_equal(x, x.astype(bool)), msg
    return True


def is_stochastic(X):
    msg = "Array must be stochastic along the columns"
    assert len(X[X < 0]) == len(X[X > 1]) == 0, msg
    assert np.allclose(np.sum(X, axis=1), np.ones(X.shape[0])), msg


def versions():
    print("[*] python Version :", sys.version)
    print("[*] np random is   :", SEED)


def set_proxy(proxy=None):
    print("[*] Proxy is       :", proxy)
    if proxy is not None:
        os.environ["http_proxy"] = proxy
        os.environ["HTTP_PROXY"] = proxy
        os.environ["https_proxy"] = proxy
        os.environ["HTTPS_PROXY"] = proxy


def check_version_proxy(proxy=None):
    print("=" * 60)
    versions()
    set_proxy(proxy=proxy)
    print("=" * 60)
