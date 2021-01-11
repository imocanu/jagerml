import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

(train_images, train_label), (test_images, test_label) = mnist.load_data()
print(train_images.shape,
      train_label.shape,
      test_images.shape,
      test_label.shape)

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

train_images = np.pad(train_images, ((0,0),(2,2),(2,2),(0,0)), 'constant')
test_images = np.pad(test_images, ((0,0),(2,2),(2,2),(0,0)), 'constant')

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(3,3), activation='relu', input_shape=(32,32,1)))
model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=120, activation='relu'))
model.add(tf.keras.layers.Dense(units=84, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

print(model.summary())
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)

log_dir = "logs/mnist"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1,
                                                      write_images=True,
                                                      write_graph=True,
                                                      embeddings_freq=1)

tf.debugging.experimental.enable_dump_debug_info(
   log_dir,
   tensor_debug_mode="FULL_HEALTH",
   circular_buffer_size=-1)

model.fit(train_images,
          train_label,
          epochs=15,
          batch_size=1024,
          validation_data=(test_images, test_label),
          callbacks=[tensorboard_callback])

plt.plot(model.history.history['accuracy'], label='Train Accuracy')
plt.plot(model.history.history['val_accuracy'], label='Test Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
# plt.savefig('accuracy.png')

plt.close()

plt.plot(model.history.history['loss'], label='Train Loss')
plt.plot(model.history.history['val_loss'], label='Test Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
# plt.savefig('loss.png')

model.evaluate(test_images, test_label)