import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import time


def mnist_dataset(batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / np.float32(255)
    y_train = y_train / np.float32(255)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(60000).repeat().batch(batch_size)
    return train_dataset, test_dataset


def mnist_dataset2(batch_size):
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    print(train_images.shape, test_images.shape)
    train_images = train_images[..., None]
    test_images = test_images[..., None]
    print(train_images.shape, test_images.shape)
    train_images = train_images / np.float32(255)
    test_images = test_images / np.float32(255)
    print(train_images.shape, test_images.shape)
    print(train_labels.shape, test_labels.shape)

    strategy = tf.distribute.MirroredStrategy()
    print("Nr of devices :", strategy.num_replicas_in_sync)

    BUFFER_SIZE = len(train_images)
    BATCH_SIZE_PER_REPLICA = 64
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(
        GLOBAL_BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

    return train_dataset, test_dataset


def build_compile_fit_cnn_model(train_dataset, test_dataset, epochs):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10))

    class PrintLR(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                              model.optimizer.lr.numpy()))

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                           save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
        PrintLR()
    ]

    history_metrics = model.fit(train_dataset,
                                epochs=epochs,
                                validation_data=test_dataset,
                                steps_per_epoch=70,
                                callbacks=callbacks)

    # Saving the model to a path on localhost.
    saved_model_path = "/tmp/tf_save"
    save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    model.save(saved_model_path, options=save_options)

    another_strategy = tf.distribute.MirroredStrategy()
    with another_strategy.scope():
        load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
        loaded = tf.keras.models.load_model(saved_model_path, options=load_options)

    return history_metrics


def plot_acc(history_metrics):
    rows = 2
    cols = 2
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    for i, metric in enumerate(history_metrics.history):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        ax.plot(history_metrics.epoch, history_metrics.history[metric])
        ax.set_title(metric)

    plt.show()


start = time.time()
batch_size = 64
train_dataset, test_dataset = mnist_dataset2(batch_size)

print("[*]TimePoint :", time.time()- start)
epochs = 15
history = build_compile_fit_cnn_model(train_dataset, test_dataset, epochs=epochs)
print("[*]TimePoint :", time.time()- start)
plot_acc(history)
