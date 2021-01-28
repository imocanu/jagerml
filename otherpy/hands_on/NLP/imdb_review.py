from imports import *


def nlp_imdb_reviews():
    datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
    print(datasets.keys())

    train_size = info.splits["train"].num_examples
    test_size = info.splits["test"].num_examples
    print(train_size, test_size)

    # for X_batch, y_batch in datasets["train"].batch(2).take(1):
    #     for review, label in zip(X_batch.numpy(), y_batch.numpy()):
    #         print("Review:", review.decode("utf-8")[:200], "...")
    #         print("Label:", label, "= Positive" if label else "= Negative")
    #         print()

    VOCAB_SIZE = 10000
    num_oov_buckets = 1000
    embed_size = 128

    def preprocess(X_batch, y_batch):
        X_batch = tf.strings.substr(X_batch, 0, 300)
        X_batch = tf.strings.regex_replace(X_batch, rb"<br\s*/?>", b" ")
        X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
        X_batch = tf.strings.split(X_batch)
        return X_batch.to_tensor(default_value=b"<pad>"), y_batch

    def create_data_set(data_type=""):
        # preprocess(X_batch, y_batch)
        vocabulary = Counter()
        for X_batch, y_batch in datasets[data_type].batch(32).map(preprocess):
            for review in X_batch:
                vocabulary.update(list(review.numpy()))

        print(len(vocabulary))

        truncated_vocabulary = [word for word, count in vocabulary.most_common()[:VOCAB_SIZE]]
        # print(truncated_vocabulary)

        word_to_id = {word: index for index, word in enumerate(truncated_vocabulary)}

        words = tf.constant(truncated_vocabulary)
        word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
        vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)

        table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

        def encode_words(X_batch, y_batch):
            return table.lookup(X_batch), y_batch

        # train_set = datasets[data_type].repeat().batch(32).map(preprocess)
        train_set = datasets[data_type].batch(32).map(preprocess)
        train_set = train_set.map(encode_words).prefetch(1)
        return train_set

    train_set = create_data_set("train")
    test_set = create_data_set("test")

    model = keras.models.Sequential([
        keras.layers.Embedding(VOCAB_SIZE + num_oov_buckets, embed_size,
                               mask_zero=True,  # not shown in the book
                               input_shape=[None]),
        keras.layers.GRU(128, return_sequences=True),
        keras.layers.GRU(128),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=["accuracy"])
    #                         steps_per_epoch=train_size // 32,
    history = model.fit(train_set,
                        epochs=5,
                        validation_data=test_set,
                        callbacks=[cb_checkpoint,
                                   cb_tensorboard,
                                   cb_early_stopping])

    plot_history(history)
    plt.show()


if __name__ == "__main__":
    check_version_proxy_gpu()
    nlp_imdb_reviews()
