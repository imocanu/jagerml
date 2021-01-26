#!/usr/bin/env python3
"""
CNN
convolutional layers and pooling layers

To reach 99.5 to 99.7% accuracy on the test set :
- add image augmentation
- batch norm
- use a learning schedule such as 1-cycle
- IF possibly, create an ensemble
"""
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds

print(tf.__version__)


dataset, info = tfds.load("fashion_mnist", as_supervised=True, with_info=True)
print(info.splits)
print(info.splits["train"])
class_names = info.features["label"].names
print(class_names)
print(dataset)
n_classes = info.features["label"].num_classes
print(n_classes)
dataset_size = info.splits["train"].num_examples
print(dataset_size)

test_set_raw, valid_set_raw, train_set_raw = tfds.load(
    "mnist",
    split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
    as_supervised=True)


# plt.figure(figsize=(12, 10))
# index = 0
# for image, label in train_set_raw.take(3):
#     index += 1
#     plt.subplot(3, 3, index)
#     plt.imshow(image)
#     plt.title("Class: {}".format(class_names[label]))
#     plt.axis("off")
# plt.show()


def central_crop(image):
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]])
    top_crop = (shape[0] - min_dim) // 4
    bottom_crop = shape[0] - top_crop
    left_crop = (shape[1] - min_dim) // 4
    right_crop = shape[1] - left_crop
    return image[top_crop:bottom_crop, left_crop:right_crop]


def random_crop(image):
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]]) * 90 // 100
    return tf.image.random_crop(image, [min_dim, min_dim, 3])


def preprocess(image, label, randomize=False):
    if randomize:
        cropped_image = random_crop(image)
        cropped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        cropped_image = central_crop(image)
    resized_image = tf.image.resize(cropped_image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label


batch_size = 32
train_set = train_set_raw.shuffle(1000)
train_set = train_set.batch(batch_size).prefetch(1)
valid_set = valid_set_raw.batch(batch_size).prefetch(1)
test_set = test_set_raw.batch(batch_size).prefetch(1)

# model = keras.models.Sequential([
#     keras.layers.Conv2D(64, 7, activation="relu", padding="same", input_shape=[224, 224, 3]),
#     keras.layers.MaxPooling2D(2),
#     keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
#     keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
#     keras.layers.MaxPooling2D(2),
#     keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
#     keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
#     keras.layers.MaxPooling2D(2),
#     keras.layers.Flatten(),
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(64, activation="relu"),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(n_classes, activation="softmax")
# ])

model = keras.models.Sequential([
    keras.layers.Conv2D(32, 7, activation="relu", padding="same", input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(63, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(n_classes, activation="softmax")
])

optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])

# history = model.fit(train_set,
#                     steps_per_epoch=int(0.75 * dataset_size / batch_size),
#                     validation_data=valid_set,
#                     validation_steps=int(0.15 * dataset_size / batch_size),
#                     epochs=5)

history = model.fit(train_set,
                    validation_data=valid_set,
                    epochs=10, batch_size=batch_size)
