from otherpy.hands_on.CNN.imports import *


def load_tfds(ds_name="mnist", ds_split=["train[:10%]", "train[:90%]"]):
    print("[*] Load tfds :", ds_name)
    (train_set_raw, test_set_raw), info = tfds.load(
        ds_name,
        split=ds_split,
        as_supervised=True,
        with_info=True)
    print("train items :", len(train_set_raw))
    print("test items  :", len(test_set_raw))
    return (train_set_raw, test_set_raw), info


def load_tfds_full(ds_name="mnist", as_supervised=True, with_info=True):
    print("[*] Load tfds :", ds_name)
    (train_set_raw, test_set_raw), info = tfds.load(ds_name,
                                                    split=['train', 'test'],
                                                    as_supervised=as_supervised,
                                                    with_info=with_info)
    return (train_set_raw, test_set_raw), info


def load_tfds_from_numpy(url="", name_ds=""):
    path = tf.keras.utils.get_file(name_ds, url)
    with np.load(path) as data:
        train_examples = data['x_train']
        train_labels = data['y_train']
        test_examples = data['x_test']
        test_labels = data['y_test']

    train_set_raw = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    test_set_raw = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

    return train_set_raw, test_set_raw


def load_tfds_from_directory():
    pass
