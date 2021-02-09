"""
https://www.tensorflow.org/guide/data

tf.data.Dataset
"""
from to_import import *


def run_test():
    train, test = tf.keras.datasets.fashion_mnist.load_data()
    check_dataset(train)
    check_dataset(test)
    images, labels = train
    check_dataset(images)
    check_dataset(labels)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    check_dataset(dataset)

    def count(stop):
        i = 0
        while i < stop:
            yield i
            i += 1

    ds_counter = tf.data.Dataset.from_generator(count,
                                                args=[25],
                                                output_types=tf.int32,
                                                output_shapes=(), )
    ds_counter = ds_counter.repeat().batch(10)
    check_dataset(ds_counter.take(5))
    # for count_batch in ds_counter.take(10):
    #     print(count_batch.numpy())

    def gen_series():
        i = 0
        while True:
            size = np.random.randint(0, 10)
            yield i, np.random.normal(size=(size,))
            i += 1

    ds_series = tf.data.Dataset.from_generator(gen_series,
                                               output_types=(tf.int32, tf.float32),
                                               output_shapes=((), (None,)))
    ds_series_batch = ds_series.shuffle(20).padded_batch(10)
    check_dataset(ds_series)
    check_dataset(ds_series_batch.take(5))


if __name__ == "__main__":
    check_version_proxy_gpu()
    run_test()
