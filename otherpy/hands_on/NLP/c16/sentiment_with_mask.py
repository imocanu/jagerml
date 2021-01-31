from to_import import *


def run_test():
    datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
    train_size = info.splits["train"].num_examples

    def preprocess(X_batch, y_batch):
        X_batch = tf.strings.substr(X_batch, 0, 300)
        X_batch = tf.strings.regex_replace(X_batch, b"<br\\s*/?>", b" ")
        X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
        X_batch = tf.strings.split(X_batch)
        return X_batch.to_tensor(default_value=b"<pad>"), y_batch

    vocabulary = Counter()
    for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
        for review in X_batch:
            vocabulary.update(list(review.numpy()))

    vocab_size = 10000
    truncated_vocabulary = [word for word, count in vocabulary.most_common()[:vocab_size]]

    print(truncated_vocabulary[:10])

    words = tf.constant(truncated_vocabulary)
    word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
    vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
    num_oov_buckets = 1000
    table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

    def encode_words(X_batch, y_batch):
        return table.lookup(X_batch), y_batch

    train_set = datasets["train"].batch(32).map(preprocess)
    train_set = train_set.map(encode_words).prefetch(1)

    EMBED_SIZE = 128
    EPOCHS = 10

    # model = keras.models.Sequential()
    # model.add(keras.layers.Embedding(vocab_size + num_oov_buckets, EMBED_SIZE, input_shape=[None]))
    # model.add(keras.layers.GRU(128, return_sequences=True))
    # model.add(keras.layers.GRU(128))
    # model.add(keras.layers.Dense(1, activation="sigmoid"))

    K = keras.backend
    inputs = keras.layers.Input(shape=[None])
    mask = keras.layers.Lambda(lambda inputs: K.not_equal(inputs, 0))(inputs)
    z = keras.layers.Embedding(vocab_size + num_oov_buckets, EMBED_SIZE)(inputs)
    z = keras.layers.LSTM(128, return_sequences=True)(z, mask=mask)
    z = keras.layers.LSTM(128)(z, mask=mask)
    outputs = keras.layers.Dense(1, activation="sigmoid")(z)
    model = keras.Model(inputs=[inputs], outputs=[outputs])

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    history = model.fit(train_set, epochs=10)

    plot_history_per_keys(history, EPOCHS)
    plt.show()
    model.save('model_sentiment.h5')


def load_model():
    datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
    train_size = info.splits["train"].num_examples

    model = tf.keras.models.load_model('model_sentiment.h5')
    model.summary()

    def preprocess(X_batch, y_batch):
        X_batch = tf.strings.substr(X_batch, 0, 300)
        X_batch = tf.strings.regex_replace(X_batch, b"<br\\s*/?>", b" ")
        X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
        X_batch = tf.strings.split(X_batch)
        return X_batch.to_tensor(default_value=b"<pad>"), y_batch

    vocabulary = Counter()
    for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
        for review in X_batch:
            vocabulary.update(list(review.numpy()))

    vocab_size = 10000
    truncated_vocabulary = [word for word, count in vocabulary.most_common()[:vocab_size]]

    words = tf.constant(truncated_vocabulary)
    word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
    vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
    num_oov_buckets = 1000
    table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)


    def encode_words():
        return table.lookup(tf.constant([b"movie not bad not bad".split()]))

    train_set = encode_words()
    predicted = model.predict(train_set)

    print("Predict:", predicted)
    if predicted > 0.5:
        print("Positive")
    else:
        print("Negative")


if __name__ == "__main__":
    check_version_proxy_gpu()
    run_test()
    # load_model()
