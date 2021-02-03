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

    print("*"*10)
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


if __name__ == "__main__":
    check_version_proxy_gpu()
    run_test()
