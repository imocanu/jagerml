"""
Glorot and He Initialization
Leaky ReLU
ELU
Scaled ELU
SELU > ELU > leaky ReLU > ReLU > tanh > logistic
For Leaky ReLU :
                keras.layers.Dense(10, kernel_initializer="he_normal"),
                keras.layers.LeakyReLU(alpha=0.2)
For SELU :
                keras.layers.Dense(10, activation="selu", kernel_initializer="lecun_normal")
Batch Normalization
AdaGrad
RMSProp
Adam ( Adamax and Nadam )
[*] Learning Rate Scheduling
"""

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print(tf.__version__)
tf.random.set_seed(42)
np.random.seed(42)

class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nLoss val/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))
        print("Acc val/train: {:.2f}".format(logs["val_sparse_categorical_accuracy"] /
                                             logs["sparse_categorical_accuracy"]))


# def exponential_decay_fn(epoch):
#     return 0.01 * 0.1 ** (epoch / 20)

def piecewise_constant_fn(epoch):
    if epoch < 2:
        print("[!]LR : 0.1")
        return 0.005
    elif epoch < 3:
        print("[!]LR : 0.05")
        return 0.0001
    else:
        print("[!]LR : 0.00001")
        return 0.0005


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)

    return exponential_decay_fn


def run_fashion_mnist():
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    X_train, X_test = X_train[:] / 255.0, X_test[:] / 255.0

    class_names = ["T-shirt/top", "Trouser", "Pullover",
                   "Dress", "Coat", "Sandal", "Shirt",
                   "Sneaker", "Bag", "Ankle boot"]

    print(class_names[y_train[0]])

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    model.add(keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"))
    model.add(keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.summary()

    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),  # RMSprop(lr=0.001, rho=0.9),
                  metrics=["accuracy"])

    # Callbacks
    run_logdir = os.path.join(os.curdir, "logs_c11")
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    checkpoint_cb = keras.callbacks.ModelCheckpoint("model_c11_cb.h5",
                                                    save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                      restore_best_weights=True)

    print_val_train_ratio_callback = PrintValTrainRatioCallback()

    #exponential_decay_fn = exponential_decay(lr0=0.01, s=20)
    #lr_scheduler = keras.callbacks.LearningRateScheduler(piecewise_constant_fn)
    #lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

    n_epochs = 25
    history = model.fit(X_train, y_train,
                        epochs=n_epochs,
                        validation_data=(X_test, y_test),
                        callbacks=[lr_scheduler])

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
    plt.show()

    plt.plot(history.epoch, history.history["lr"], "bo-")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate", color='b')
    plt.tick_params('y', colors='b')
    plt.gca().set_xlim(0, n_epochs - 1)
    plt.grid(True)
    ax2 = plt.gca().twinx()
    ax2.plot(history.epoch, history.history["val_loss"], "r^-")
    ax2.set_ylabel('Validation Loss', color='r')
    ax2.tick_params('y', colors='r')

    plt.title("Reduce LR on Plateau", fontsize=14)
    plt.show()

    model.evaluate(X_test, y_test)

    print("Save model ...")
    model.save("model_c11.h5")
    print("Load model ...")
    model = keras.models.load_model("model_c11.h5")

    X_new = X_test[:3]
    y_proba = model.predict(X_new)
    print(y_proba.round(2))

    # y_pred = model.predict_classes(X_new)  predict_classes is deprecated
    y_pred = np.argmax(model.predict(X_new), axis=-1)
    print("Pred labels :", np.array(class_names)[y_pred])

    y_new = y_test[:3]
    print("Final       :", np.array(class_names)[y_new])

    # test_logdir = run_logdir
    # writer = tf.summary.create_file_writer(test_logdir)
    # with writer.as_default():
    #     for step in range(1, 1000 + 1):
    #         tf.summary.scalar("my_scalar", np.sin(step / 10), step=step)
    #         data = (np.random.randn(100) + 2) * step / 100  # some random data
    #         tf.summary.histogram("my_hist", data, buckets=50, step=step)
    #         images = np.random.rand(2, 32, 32, 3)  # random 32×32 RGB images
    #         tf.summary.image("my_images", images * step / 1000, step=step)
    #         texts = ["The step is " + str(step), "Its square is " + str(step ** 2)]
    #         tf.summary.text("my_text", texts, step=step)
    #         sine_wave = tf.math.sin(tf.range(12000) / 48000 * 2 * np.pi * step)
    #         audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1])
    #         tf.summary.audio("my_audio", audio, sample_rate=48000, step=step)


if __name__ == "__main__":
    run_fashion_mnist()
