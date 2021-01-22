from otherpy.hands_on.CNN.imports import *


def cnn_model(labels=1, img_size=28, img_channels=1):

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, 7, activation="relu", padding="same",
                                  input_shape=(img_size, img_size, img_channels)))
    model.add(keras.layers.MaxPooling2D(2))
    model.add(keras.layers.Conv2D(63, 3, activation="relu", padding="same"))
    model.add(keras.layers.Conv2D(64, 3, activation="relu", padding="same"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(labels, activation="softmax"))

    return model
