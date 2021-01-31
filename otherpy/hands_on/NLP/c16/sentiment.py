from to_import import *


def run_test():
    # (X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data()
    #
    # word_index = keras.datasets.imdb.get_word_index()
    # id_to_word = {id_ + 3: word for word, id_ in word_index.items()}
    # for id_, token in enumerate(("<pad>", "<sos>", "<unk>")):
    #     id_to_word[id_] = token
    #
    # print(" ".join([id_to_word[id_] for id_ in X_train[0][:30]]))
    # print(" ".join([id_to_word[id_] for id_ in X_train[1][:30]]))

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

    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(vocab_size + num_oov_buckets, EMBED_SIZE, input_shape=[None]))
    model.add(keras.layers.GRU(128, return_sequences=True))
    model.add(keras.layers.GRU(128))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    history = model.fit(train_set, epochs=10)

    plot_history_per_keys(history, EPOCHS)
    plt.show()
    model.save('model_sentiment.h5')


def load_model():
    model = tf.keras.models.load_model('model_sentiment.h5')
    model.summary()

    # def preprocess(texts):
    #     X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    #     return tf.one_hot(X, MAX_ID)

    def preprocess(X_to_predict):
        X_to_predict = tf.strings.substr(X_to_predict, 0, 300)
        X_to_predict = tf.strings.regex_replace(X_to_predict, b"<br\\s*/?>", b" ")
        X_to_predict = tf.strings.regex_replace(X_to_predict, b"[^a-zA-Z']", b" ")
        X_to_predict = tf.strings.split(X_to_predict)
        return X_to_predict.to_tensor(default_value=b"<pad>")

    # def next_char(text, temperature=1):
    #     X_new = preprocess([text])
    #     y_proba = model.predict(X_new)[0, -1:, :]
    #     rescaled_logits = tf.math.log(y_proba) / temperature
    #     char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    #     return tokenizer.sequences_to_texts(char_id.numpy())[0]
    #
    # def complete_text(text, n_chars=50, temperature=1):
    #     for _ in range(n_chars):
    #         text += next_char(text, temperature)
    #     return text


if __name__ == "__main__":
    check_version_proxy_gpu()
    run_test()
    # load_model()
