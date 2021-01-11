import tensorflow as tf
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# BASE_PATH = '/home/imocanu/Downloads/dogs-vs-cats/train/'
TRAIN_PATH = '/home/imocanu/Downloads/dogs_cats_v2/dataset/training_set'
VAL_PATH = '/home/imocanu/Downloads/dogs_cats_v2/dataset/test_set'
batch_size = 32
epochs = 60
IMG_HEIGHT = 150
IMG_WIDTH = 150

# train_image_generator = ImageDataGenerator(rescale=1. / 255,
#                                            rotation_range=45,
#                                            width_shift_range=.15,
#                                            height_shift_range=.15,
#                                            horizontal_flip=True,
#                                            zoom_range=0.3)
#
# validation_image_generator = ImageDataGenerator(rescale=1. / 255)

train_data_gen = tf.keras.preprocessing.image_dataset_from_directory(batch_size=batch_size,
                                                                     directory=TRAIN_PATH,
                                                                     shuffle=True,
                                                                     seed=143,
                                                                     labels='inferred',
                                                                     image_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                     label_mode='categorical')

val_data_gen = tf.keras.preprocessing.image_dataset_from_directory(batch_size=batch_size,
                                                                   directory=VAL_PATH,
                                                                   seed=143,
                                                                   image_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                   labels='inferred',
                                                                   label_mode='categorical')


def create_model():
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu',
               input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(128, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(2, activation='softmax')])
    return model


log_dir = "logs/fit"

# Create and Compile the model
model = create_model()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1,
                                                      write_images=True)

model.fit_generator(
    train_data_gen,
    epochs=epochs,
    validation_data=val_data_gen,
    callbacks=[tensorboard_callback]
)
