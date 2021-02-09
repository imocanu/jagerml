"""
https://www.tensorflow.org/guide/data

tf.keras.preprocessing.image.ImageDataGenerator
"""
from to_import import *


def run_test():
    flowers = tf.keras.utils.get_file(
        'flower_photos',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        untar=True)

    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, rotation_range=20)
    images, labels = next(img_gen.flow_from_directory(flowers))
    check_dataset(images)
    check_dataset(labels)

    ds = tf.data.Dataset.from_generator(
        lambda: img_gen.flow_from_directory(flowers),
        output_types=(tf.float32, tf.float32),
        output_shapes=([32, 256, 256, 3], [32, 5])
    )

    check_dataset(ds)

    for images, label in ds.take(1):
        print('images.shape: ', images.shape)
        print('labels.shape: ', labels.shape)


if __name__ == "__main__":
    check_version_proxy_gpu()
    run_test()
