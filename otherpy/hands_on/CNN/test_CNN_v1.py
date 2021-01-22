#!/usr/bin/env python3

from otherpy.hands_on.CNN.imports import *
from otherpy.hands_on.CNN.load_ds import *
from otherpy.hands_on.CNN.ml import *


def run_cnn():
    (train_ds, test_ds), info = load_tfds_full()

    label_counter, train_counter, test_counter = check_ds_info_return_counters(info)
    check_ds_images_shape(train_ds)

    IMG_SIZE = 28
    IMG_CHANNEL = 1
    BATCH_SIZE = 32
    EPOCHS = 5
    LABELS = label_counter

    # check_ds_show_images(train_ds)
    train_set, test_set = prep_rescale_without_batch(train_ds,
                                                     test_ds)

    # check_ds_show_images(train_set)
    check_ds_images_shape(test_set)

    train_set, test_set = prep_resize_rescale(train_ds,
                                              test_ds,
                                              img_size=IMG_SIZE,
                                              batch_size=BATCH_SIZE)

    model = cnn_model(labels=LABELS, img_size=IMG_SIZE, img_channels=IMG_CHANNEL)
    model.summary()

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="nadam",
                  metrics=["accuracy"])

    history = model.fit(train_set,
                        validation_data=test_set,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE)


def run_cnn_v2():
    (train_ds, test_ds), info = load_tfds(ds_name="tf_flowers")
    label_counter = check_ds_info_return_classes(info)
    check_ds_images_shape(train_ds)

    IMG_SIZE = 200
    IMG_CHANNEL = 3
    BATCH_SIZE = 32
    EPOCHS = 5
    LABELS = label_counter

    check_ds_show_images(train_ds)
    train_set, test_set = prep_rescale_without_batch(train_ds,
                                                     test_ds)

    check_ds_show_images(train_set)
    check_ds_images_shape(test_set)

    train_set, test_set = prep_resize_rescale(train_ds,
                                              test_ds,
                                              img_size=IMG_SIZE,
                                              batch_size=BATCH_SIZE)

    model_cnn = cnn_model(labels=LABELS, img_size=IMG_SIZE, img_channels=IMG_CHANNEL)
    model_cnn.summary()

    model_cnn.compile(loss="sparse_categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

    history = model_cnn.fit(train_set,
                            validation_data=test_set,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE)


if __name__ == "__main__":
    check_other_gpu()
    versions()
    # run_cnn()
    run_cnn_v2()
