from otherpy.hands_on.CNN.imports import *


def prep_resize_rescale(train_ds, test_ds, img_size=28, batch_size=32):
    def resize_and_rescale(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [img_size, img_size])
        image = (image / 255.0)
        return image, label

    train_set = train_ds.shuffle(1000).map(resize_and_rescale).batch(batch_size).prefetch(1)
    valid_set = test_ds.shuffle(1000).map(resize_and_rescale).batch(batch_size).prefetch(1)

    return train_set, valid_set


def prep_reshape(image, shape=[-1, 28, 28, 1]):
    return tf.reshape(image, shape=shape)


def prep_rescale_reshape(train_ds, test_ds, batch_size=32, shape=[-1, 28, 28, 1]):
    def resize_and_rescale(image, label):
        image = prep_reshape(image, shape=shape)
        image = tf.cast(image, tf.float32)
        image = (image / 255.0)
        return image, label

    train_set = train_ds.shuffle(100).map(resize_and_rescale).batch(batch_size).prefetch(1)
    valid_set = test_ds.shuffle(100).map(resize_and_rescale).batch(batch_size).prefetch(1)

    return train_set, valid_set


def prep_rescale_only(train_ds, test_ds, batch_size=32):
    def resize_and_rescale(image, label):
        image = tf.cast(image, tf.float32)
        image = (image / 255.0)
        return image, label

    train_set = train_ds.shuffle(100).map(resize_and_rescale).batch(batch_size).prefetch(1)
    valid_set = test_ds.shuffle(100).map(resize_and_rescale).batch(batch_size).prefetch(1)

    return train_set, valid_set


def prep_rescale_without_batch(train_ds, test_ds):
    def resize_and_rescale(image, label):
        image = tf.cast(image, tf.float32)
        image = (image / 255.0)
        return image, label

    train_set = train_ds.shuffle(100).map(resize_and_rescale)
    valid_set = test_ds.shuffle(100).map(resize_and_rescale)

    return train_set, valid_set
