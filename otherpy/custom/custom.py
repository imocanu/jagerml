import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print(tf.test.is_gpu_available())


def time_matmul(xt):
    start = time.time()
    for loop in range(10):
        tf.matmul(xt, xt)

    result = time.time() - start

    print("10 loops: {:0.2f}ms".format(1000 * result))


# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)

DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
with np.load(path) as data:
    train_examples = data['x_train']
    train_labels = data['y_train']
    test_examples = data['x_test']
    test_labels = data['y_test']
print(type(train_examples))

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
print(type(train_dataset))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
print(type(train_dataset))


class DenseCustom(tf.keras.layers.Layer):
    def __init__(self, units=10):
        super(DenseCustom, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.units])

    def call(self, input):
        return tf.matmul(input, self.kernel)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(DenseCustom(10))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_dataset,
                    epochs=20,
                    validation_data=test_dataset,
                    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2))

metrics = history.history

# plt.plot(history.epoch, metrics['loss'], metrics['accuracy'])
# plt.legend(['loss', 'accuracy'])
# plt.show()

rows = 2
cols = 2
n = rows * cols
fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
for i, metric in enumerate(history.history):
    r = i // cols
    c = i % cols
    print(r, c)
    ax = axes[r][c]
    ax.plot(history.epoch, history.history[metric])
    ax.set_title(metric)

plt.show()

model.evaluate(test_dataset)
