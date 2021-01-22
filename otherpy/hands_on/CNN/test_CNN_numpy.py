#!/usr/bin/env python3

from otherpy.hands_on.CNN.imports import *
from otherpy.hands_on.CNN.load_ds import *
from otherpy.hands_on.CNN.ml import *


def run_cnn():
    URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    NAME_DS = "mnist.npz"
    train_ds, test_ds = load_tfds_from_numpy(url=URL,
                                             name_ds=NAME_DS)

    # !!! label_counter, train_counter, test_counter = check_ds_info_return_classes(info)
    check_ds_images_shape(train_ds)

    IMG_SIZE = 28
    IMG_CHANNEL = 1
    BATCH_SIZE = 32
    EPOCHS = 5
    LABELS = 10
    SHAPE = [28, 28, 1]

    # check_ds_show_images(train_ds)
    train_set, test_set = prep_rescale_without_batch(train_ds,
                                                     test_ds)

    # check_ds_show_images(train_set)
    check_ds_images_shape(test_set)

    train_set, test_set = prep_rescale_reshape(train_ds,
                                               test_ds,
                                               batch_size=BATCH_SIZE,
                                               shape=SHAPE)

    check_ds_images_shape(test_set)

    model_cnn = cnn_model(labels=LABELS, img_size=IMG_SIZE, img_channels=IMG_CHANNEL)
    model_cnn.summary()

    model_cnn.compile(loss="sparse_categorical_crossentropy",
                      optimizer="nadam",
                      metrics=["accuracy"])

    history = model_cnn.fit(train_set,
                            validation_data=test_set,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE)

    loss, accuracy = model_cnn.evaluate(train_set)
    print("Loss     :", loss)
    print("Accuracy :", accuracy)
    plot_fig(history, EPOCHS)


def run_cnn_v2():
    pass


if __name__ == "__main__":
    check_other_gpu()
    versions()
    run_cnn()
    # run_cnn_v2()
