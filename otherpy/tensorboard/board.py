import tensorflow as tf
import numpy as np

data_size = 1000
# 80% of the data is for training.
train_pct = 0.8

train_size = int(data_size * train_pct)

# Create some input data between -1 and 1 and randomize it.
x = np.linspace(-1, 1, data_size)
np.random.shuffle(x)

# Generate the output data.
# y = 0.5x + 2 + noise
y = 0.5 * x + 2 + np.random.normal(0, 0.05, (data_size,))

# Split into test and train pairs.
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

# logdir = "logs/scalars/"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
#
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(16, input_dim=1),
#     tf.keras.layers.Dense(1),
# ])
#
# model.compile(
#     loss='mse', # keras.losses.mean_squared_error
#     optimizer=tf.keras.optimizers.SGD(lr=0.2),
# )
#
# training_history = model.fit(
#     x_train, # input
#     y_train, # output
#     batch_size=train_size,
#     verbose=0, # Suppress chatty output; use Tensorboard instead
#     epochs=100,
#     validation_data=(x_test, y_test),
#     callbacks=[tensorboard_callback],
# )
#
# print("Average test loss: ", np.average(training_history.history['loss']))


logdir = "logs/scalars/"
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()


def lr_schedule(epoch):
    """
  Returns a custom learning rate that decreases as epochs progress.
  """
    learning_rate = 0.2
    if epoch > 10:
        learning_rate = 0.02
    if epoch > 20:
        learning_rate = 0.01
    if epoch > 50:
        learning_rate = 0.005

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, input_dim=1),
    tf.keras.layers.Dense(1),
])

model.compile(
    loss='mse',  # keras.losses.mean_squared_error
    optimizer=tf.keras.optimizers.SGD(),
)

training_history = model.fit(
    x_train,  # input
    y_train,  # output
    batch_size=train_size,
    verbose=0,  # Suppress chatty output; use Tensorboard instead
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback, lr_callback],
)
