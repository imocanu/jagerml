#!/usr/bin/env python3

from otherpy.hands_on.CNN.imports import *
from otherpy.hands_on.CNN.load_ds import *
from otherpy.hands_on.CNN.ml import *


def run_cnn():
    URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    DIRECTORY = "flower_photos"
    data_dir = load_tfds_from_url_directory(url=URL,
                                            name_ds=DIRECTORY)

    # !!! label_counter, train_counter, test_counter = check_ds_info_return_classes(info)

    IMG_SIZE = 150
    IMG_CHANNEL = 3
    BATCH_SIZE = 32
    EPOCHS = 5
    SHAPE = [28, 28, 1]

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE)

    check_ds_images_shape(train_ds)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE)

    LABELS = len(train_ds.class_names)  # ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    print(LABELS)
    print("TOTAL train :", len(train_ds)*BATCH_SIZE)
    print("TOTAL test  :",len(test_ds)*BATCH_SIZE)

    # check_ds_show_batch_images(train_ds)
    # check_ds_images_shape(test_set)

    train_set, test_set = prep_rescale_only(train_ds,
                                            test_ds,
                                            batch_size=BATCH_SIZE)

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
