from to_import import *


def run_test():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    fname = "aclImdb_v1.tar.gz"
    path_to_zip = tf.keras.utils.get_file(fname=fname, origin=url, extract=True)

    main_dir = os.path.join(os.path.dirname(path_to_zip), 'aclImdb')
    train_dir = os.path.join(main_dir, 'train')
    print("*", os.listdir(train_dir))
    remove_dir = os.path.join(train_dir, 'unsup')
    import shutil
    shutil.rmtree(remove_dir)

    print(os.listdir(train_dir))

    test_ds = tf.keras.preprocessing.text_dataset_from_directory(directory=train_dir)

    # dataset = tf.data.Dataset.list_files(train_path)
    print(len(test_ds))

    pos_abs_paths = []
    pos_path = os.path.join(train_dir, 'pos')
    print("POS :", pos_path)
    # print("POS :", os.path.abspath(os.listdir(pos_path)))
    for f in os.listdir(pos_path):
        q = pos_path + "/" + f
        pos_abs_paths.append(q)
    test_ds_textline = tf.data.TextLineDataset(pos_abs_paths)
    for f in test_ds_textline.take(1):
        print(f)

    print("*" * 10)
    files_ds = tf.data.Dataset.from_tensor_slices(pos_abs_paths)
    lines_ds = files_ds.interleave(tf.data.TextLineDataset, cycle_length=3)

    for i, line in enumerate(lines_ds.take(2)):
        if i % 3 == 0:
            print()
        print(line.numpy())

    titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
    titanic_lines = tf.data.TextLineDataset(titanic_file)
    for line in titanic_lines.take(10):
        print(line.numpy())

    # from Numpy
    train, test = tf.keras.datasets.fashion_mnist.load_data()
    images, labels = train
    images = images / 255
    print(type(images), type(labels))
    print(len(images), len(labels))
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    # for x, y in dataset.take(2):
    #    print(x, y)

    def count(stop):
        i = 0
        while i < stop:
            yield i
            i += 1

    flowers = tf.keras.utils.get_file(
        'flower_photos',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        untar=True)

    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, rotation_range=20)
    ds = tf.data.Dataset.from_generator(
        lambda: img_gen.flow_from_directory(flowers),
        output_types=(tf.float32, tf.float32),
        output_shapes=([32, 256, 256, 3], [32, 5])
    )

    for images, label in ds.take(1):
        print(images.shape, label.shape)


if __name__ == "__main__":
    check_version_proxy_gpu()
    run_test()
