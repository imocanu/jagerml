#!/usr/bin/env python3
"""
CNN
convolutional layers and pooling layers
"""
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print(tf.__version__)

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# from sklearn.datasets import load_sample_image
# from PIL import Image
#
# image = Image.open("jagerML.jpg")
#
# # load and display an image with Matplotlib
# from matplotlib import image
# from matplotlib import pyplot
#
# # load image as pixel array
# data_bg = image.imread('ml_bg.jpg')
# data_fg = image.imread('ml_bg.jpg')
# data_bg = data_bg / 255
# data_fg = data_fg / 255
#
# images = np.array([data_bg, data_fg])
# batch_size, height, width, channels = images.shape
# # Create 2 filters
# filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
# filters[:, 3, :, 0] = 1  # vertical line
# filters[3, :, :, 1] = 1  # horizontal line
#
# outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")
#
# plt.imshow(outputs[0, :, :, 1], cmap="gray")  # plot 1st image's 2nd feature map
# plt.show()

import tensorflow_datasets as tfds

dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
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
train_set = train_set.map(preprocess).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)

# base_model = keras.applications.xception.Xception(weights="imagenet",
#                                                   include_top=False)
# avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
# output = keras.layers.Dense(n_classes, activation="softmax")(avg)
# model = keras.models.Model(inputs=base_model.input, outputs=output)

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
    keras.layers.Conv2D(64, 7, activation="relu", padding="same", input_shape=[224, 224, 3]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(n_classes, activation="softmax")
])

# model = tf.keras.models.Sequential(layers=[
#     tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(224, 224, 3)),
#     tf.keras.layers.MaxPool2D(),
#     tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPool2D((4, 4)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
# ],
#     name='ConvModel')

# model.compile(optimizer="adam",
#               loss_weights="categorical_crossentropy",
#               metrics=["accuracy"])
#
# model.fit(train_set,
#           steps_per_epoch=int(0.75 * dataset_size / batch_size),
#           validation_data=valid_set,
#           validation_steps=int(0.15 * dataset_size / batch_size),
#           epochs=5)

# for layer in base_model.layers:
#     layer.trainable = False

optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
# history = model.fit(train_set,
#                     steps_per_epoch=int(0.75 * dataset_size / batch_size),
#                     validation_data=valid_set,
#                     validation_steps=int(0.15 * dataset_size / batch_size),
#                     epochs=5)

history = model.fit(train_set,
                    validation_data=valid_set,
                    epochs=10 , batch_size=batch_size)

