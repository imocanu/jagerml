from otherpy.hands_on.CNN.imports import *


def load_tfds(ds_name="mnist", ds_split=["train[:10%]", "train[:90%]"]):
    print("[*] Load tfds :", ds_name)
    (train_set_raw, test_set_raw), info = tfds.load(
        ds_name,
        split=ds_split,
        as_supervised=True,
        with_info=True)
    print("train items :", len(train_set_raw))
    print("test items  :", len(test_set_raw))
    return (train_set_raw, test_set_raw), info


def load_tfds_full(ds_name="mnist", as_supervised=True, with_info=True):
    print("[*] Load tfds :", ds_name)
    (train_set_raw, test_set_raw), info = tfds.load(ds_name,
                                                    split=['train', 'test'],
                                                    as_supervised=as_supervised,
                                                    with_info=with_info)
    return (train_set_raw, test_set_raw), info


def load_tfds_from_numpy(url="", name_ds=""):
    path = tf.keras.utils.get_file(name_ds, url)
    with np.load(path) as data:
        train_examples = data['x_train']
        train_labels = data['y_train']
        test_examples = data['x_test']
        test_labels = data['y_test']

    train_set_raw = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    test_set_raw = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

    return train_set_raw, test_set_raw


def load_tfds_from_url_directory(url="", name_ds="images"):
    import pathlib

    # dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(origin=url,
                                       fname=name_ds,
                                       untar=True)
    data_dir = pathlib.Path(data_dir)
    return data_dir


def load_tfds_from_data_dir(data_dir):
    image_count = len(list(data_dir.glob('*/*.jpg')))
    img_height = 180
    img_width = 180
    batch_size = 32
    AUTOTUNE = tf.data.AUTOTUNE

    print(image_count)
    list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
    print(class_names)

    for f in list_ds.take(5):
        print(f.numpy())

    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == class_names
        # Integer encode the label
        return tf.argmax(one_hot)

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # resize the image to the desired size
        img = tf.image.resize(img, [img_height, img_width])
        img = tf.cast(img, tf.float32)
        img = (img / 255.0)
        return img

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    test_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    for image, label in train_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

    def configure_for_performance(ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)

    image_batch, label_batch = next(iter(train_ds))

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        label = label_batch[i]
        plt.title(class_names[label])
        plt.axis("off")

    return train_ds, val_ds
