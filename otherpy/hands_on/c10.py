"""
ANN = Artificial Neural Network
MLP = Multilayer Perceptron
||
V
SVM = Support Vector Machine
Perceptron = simplest ANN and represent a single layer of TLU
Dense layer (connected layer) = all neurons from a layer are connected
to every neurons from previous layer

Huber loss = MSE + MAE
kernel = matrix of connection weights or bias_initializer when creating the layer
"""

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print(tf.__version__)


def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[28, 28]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(10))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model


class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nLoss val/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))
        print("Acc val/train: {:.2f}".format(logs["val_sparse_categorical_accuracy"] /
                                             logs["sparse_categorical_accuracy"]))


"""
TO ADD :        on_epoch_begin(), on_epoch_end()
                on_batch_begin(), on_batch_end()
                on_train_begin(), on_train_end(), 
(by evaluate)   on_test_begin(), on_test_end(), on_test_batch_begin(), on_test_batch_end()
(by predict )   on_predict_begin(), on_predict_end(), on_predict_batch_begin(), on_predict_batch_end() 
"""


def run_fine_tuning_fashion_mnist():
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    X_train, X_test = X_train[:] / 255.0, X_test[:] / 255.0

    class_names = ["T-shirt/top", "Trouser", "Pullover",
                   "Dress", "Coat", "Sandal", "Shirt",
                   "Sneaker", "Bag", "Ankle boot"]

    print(class_names[y_train[0]])

    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

    keras_reg.fit(X_train, y_train, epochs=100,
                  validation_data=(X_test, y_test),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])

    mse_test = keras_reg.score(X_test, y_test)
    X_new = X_test[:3]
    y_pred = np.argmax(keras_reg.predict(X_new), axis=-1)
    print("\nMSE_test     :", mse_test)
    print("Pred labels :", np.array(class_names)[y_pred])
    y_new = y_test[:3]
    print("Final       :", np.array(class_names)[y_new])

    from scipy.stats import reciprocal
    from sklearn.model_selection import RandomizedSearchCV

    param_distribs = {
        "n_hidden": [0, 1, 2, 3],
        "n_neurons": np.arange(1, 100),
        "learning_rate": reciprocal(3e-4, 3e-2),
    }

    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
    rnd_search_cv.fit(X_train, y_train, epochs=100,
                      validation_data=(X_test, y_test),
                      callbacks=[keras.callbacks.EarlyStopping(patience=10)])

    print("Best params :", rnd_search_cv.best_params_)
    print("Best score  :", rnd_search_cv.best_score_)
    final_model = rnd_search_cv.best_estimator_.model
    final_model.summary()


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
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.summary()

    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(),
                  metrics=[keras.metrics.sparse_categorical_accuracy])

    # Callbacks
    run_logdir = os.path.join(os.curdir, "logs_c10")
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    checkpoint_cb = keras.callbacks.ModelCheckpoint("model_c10_cb.h5",
                                                    save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                      restore_best_weights=True)
    print_val_train_ratio_callback = PrintValTrainRatioCallback()

    history = model.fit(X_train, y_train,
                        epochs=20,
                        validation_data=(X_test, y_test),
                        callbacks=[tensorboard_cb,
                                   checkpoint_cb,
                                   early_stopping_cb,
                                   print_val_train_ratio_callback])

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
    plt.show()

    model.evaluate(X_test, y_test)

    print("Save model ...")
    model.save("model_c10.h5")
    print("Load model ...")
    model = keras.models.load_model("model_c10.h5")

    X_new = X_test[:3]
    y_proba = model.predict(X_new)
    print(y_proba.round(2))

    # y_pred = model.predict_classes(X_new)  predict_classes is deprecated
    y_pred = np.argmax(model.predict(X_new), axis=-1)
    print("Pred labels :", np.array(class_names)[y_pred])

    y_new = y_test[:3]
    print("Final       :", np.array(class_names)[y_new])

    test_logdir = run_logdir
    writer = tf.summary.create_file_writer(test_logdir)
    with writer.as_default():
        for step in range(1, 1000 + 1):
            tf.summary.scalar("my_scalar", np.sin(step / 10), step=step)
            data = (np.random.randn(100) + 2) * step / 100  # some random data
            tf.summary.histogram("my_hist", data, buckets=50, step=step)
            images = np.random.rand(2, 32, 32, 3)  # random 32×32 RGB images
            tf.summary.image("my_images", images * step / 1000, step=step)
            texts = ["The step is " + str(step), "Its square is " + str(step ** 2)]
            tf.summary.text("my_text", texts, step=step)
            sine_wave = tf.math.sin(tf.range(12000) / 48000 * 2 * np.pi * step)
            audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1])
            tf.summary.audio("my_audio", audio, sample_rate=48000, step=step)


if __name__ == "__main__":
    # run_fashion_mnist()
    run_fine_tuning_fashion_mnist()
