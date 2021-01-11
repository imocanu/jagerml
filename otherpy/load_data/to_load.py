#!/usr/bin/env python3

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pathlib

'''
https://console.cloud.google.com/storage/browser/tfds-data
'''
flowers_dataset_name = "flower_photos"
flowers_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

iris_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
titanic_dataset_file = tf.keras.utils.get_file("train.csv",
                                               "https://storage.googleapis.com/tf-datasets/titanic/train.csv")

train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"],
                                  batch_size=-1, as_supervised=True)

train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)

# tf.keras.applications.
# https://www.tensorflow.org/guide/estimator
# import tensorflow_datasets
# tf.keras.datasets.fashion_mnist
# tf.keras.datasets.cifar10.load_data()

# tf.nn.....

def get_data_dir(dataset_url, dataset_name):
    data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                       fname=dataset_name,
                                       untar=True)
    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print("Img count :", image_count)

    roses = list(data_dir.glob('roses/*'))
    im = PIL.Image.open(str(roses[0]))
    width, height = im.size
    print("Original size : ", width, height)

    img_height = 180
    img_width = 180

    return data_dir, ( img_height, img_width )


def create_dataset(dataset_data_dir, img_height=180, img_width=180):
    batch_size = 32
    #img_height = 320
    #img_width = 238
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=dataset_data_dir,
                                                                   validation_split=0.2,
                                                                   subset="training",
                                                                   seed=356,
                                                                   image_size=(img_height, img_width),
                                                                   batch_size=batch_size)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=dataset_data_dir,
                                                                  validation_split=0.2,
                                                                  subset="validation",
                                                                  seed=356,
                                                                  image_size=(img_height, img_width),
                                                                  batch_size=batch_size)

    train_class_names = train_ds.class_names
    print(type(train_ds), len(train_ds), train_class_names)
    print(train_ds)
    test_class_names = test_ds.class_names
    print(type(test_ds), len(test_ds), test_class_names)
    print(test_ds)

    return (train_ds, test_ds), (train_class_names, test_class_names)
