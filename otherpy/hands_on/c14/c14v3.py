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

(valid_set_raw, train_set_raw), info = tfds.load(
    "tf_flowers",
    split=["train[10%:25%]", "train[25%:]"],
    as_supervised=True,
    with_info=True)

print(info.splits)
print(info.splits["train"])
class_names = info.features["label"].names
print(class_names)
n_classes = info.features["label"].num_classes
print(n_classes)
dataset_size = info.splits["train"].num_examples
print(dataset_size)
print(type(valid_set_raw), valid_set_raw)

# train_set = train_set_raw.shuffle(1000)
print(type(train_set_raw), train_set_raw)

IMG_SIZE = 150
batch_size = 32


def resize_and_rescale(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = (image / 255.0)
    return image, label


def prep(image, label):
    # image = tf.cast(image, tf.float32)
    resized_image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    # prep_image = (resized_image / 255.0)
    # final_image = keras.applications.xception.preprocess_input(prep_image)
    return resized_image, label


if hasattr(train_set_raw, "map"):
    print("OKKK - RAW")

train_set = train_set_raw.shuffle(1000).map(resize_and_rescale).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.shuffle(1000).map(resize_and_rescale).batch(batch_size).prefetch(1)

if hasattr(train_set, "map"):
    print("OKKK - SET")

for image, label in train_set.take(3):
    print(type(image), label)
    print(image.shape)

train_set_prev = train_set_raw.shuffle(1000).map(resize_and_rescale)
plt.figure(figsize=(12, 10))
index = 0
for image, label in train_set_prev.take(9).cache():
    index += 1
    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.title("Class: {}".format(class_names[label]))
    plt.axis("off")
    print("-> img shape : ", image.shape)


# plt.show()
# exit()


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


def augment(image_label, seed):
    image, label = image_label
    image, label = resize_and_rescale(image, label)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
    # Make a new seed
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    # Random crop back to the original size
    image = tf.image.stateless_random_crop(
        image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
    # Random brightness
    image = tf.image.stateless_random_brightness(
        image, max_delta=0.5, seed=new_seed)
    image = tf.clip_by_value(image, 0, 1)
    return image, label


resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
])

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

data_normalization = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Normalization(),
])

# train_set = train_set_raw.shuffle(1000)
# train_set1 = train_set_raw.resize(IMG_SIZE, IMG_SIZE)

# train_set = train_set.batch(batch_size).prefetch(1)
# valid_set = valid_set_raw.batch(batch_size).prefetch(1)
# test_set = test_set_raw.batch(batch_size).prefetch(1)

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

model = keras.models.Sequential()
# model.add(data_normalization)
# model.add(resize_and_rescale)
# model.add(data_augmentation)
# model.add(keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE))
# model.add(keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(keras.layers.Conv2D(32, 7, activation="relu", padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(63, 3, activation="relu", padding="same"))
model.add(keras.layers.Conv2D(64, 3, activation="relu", padding="same"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(n_classes, activation="softmax"))

# optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="nadam",
              metrics=["accuracy"])

# history = model.fit(train_set,
#                     steps_per_epoch=int(0.75 * dataset_size / batch_size),
#                     validation_data=valid_set,
#                     validation_steps=int(0.15 * dataset_size / batch_size),
#                     epochs=5)

EPOCHS = 5
history = model.fit(train_set,
                    validation_data=valid_set,
                    epochs=EPOCHS,
                    batch_size=batch_size)


# model.evaluate(train_set, test_set)

def plot_fig(i, history_model):
    fig = plt.figure()
    plt.plot(range(1, EPOCHS + 1), history_model.history['val_accuracy'], label='validation')
    plt.plot(range(1, EPOCHS + 1), history_model.history['acc'], label='training')
    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.xlim([1, EPOCHS])
    #     plt.ylim([0,1])
    plt.grid(True)
    plt.title("Model Accuracy")
    plt.show()
    # fig.savefig('img/'+str(i)+'-accuracy.jpg')
    # plt.close(fig)


plot_fig(1, history)
