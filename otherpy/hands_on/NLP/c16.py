from imports import *
from tqdm import tqdm


class ResetStatesCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])


def plot_history(history):
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    plt.ylim(0, None)


def plot_history_without_val(history):
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    metric = 'accuracy'
    plt.plot(history.history[metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.ylim(None, 1)

    plt.subplot(1, 2, 2)
    metric = 'loss'
    plt.plot(history.history[metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.ylim(0, None)


def run_test1():
    url = "https://homl.info/shakespeare"
    fname = "shakespeare.txt"
    file_path = keras.utils.get_file(fname, url)

    with open(file_path) as f:
        shakespeare_text = f.read(100000)

    # print(shakespeare)
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(shakespeare_text)

    # print(tokenizer.texts_to_sequences(["First"]))
    max_id = len(tokenizer.word_index)
    print("[*] Number of unique words :", len(tokenizer.word_index))
    print("[*] Number of total words  :", tokenizer.document_count)

    [encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1

    train_size = tokenizer.document_count * 90 // 100
    n_steps = 100
    batch_size = 32
    window_length = n_steps + 1  # target = input shifted 1 character ahead

    def preprocess(texts):
        X = np.array(tokenizer.texts_to_sequences(texts)) - 1
        return tf.one_hot(X, max_id)

    # dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
    #
    # dataset = dataset.window(window_length, shift=1, drop_remainder=True)
    #
    # dataset = dataset.flat_map(lambda window: window.batch(window_length))
    #
    # dataset = dataset.shuffle(10000).batch(batch_size)
    # dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
    #
    # dataset = dataset.map(
    #     lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
    #
    # dataset = dataset.prefetch(1)
    #
    # model = keras.models.Sequential([
    #     keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id],
    #                      dropout=0.2, recurrent_dropout=0.2),
    #     keras.layers.GRU(128, return_sequences=True,
    #                      dropout=0.2, recurrent_dropout=0.2),
    #     keras.layers.TimeDistributed(keras.layers.Dense(max_id,
    #                                                     activation="softmax"))
    # ])
    # model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    # history = model.fit(dataset, epochs=10)
    #
    #
    # X_new = preprocess(["How are yo"])
    # Y_pred = np.argmax(model.predict(X_new), axis=-1)  # DEPRECATETD model.predict_classes(X_new)
    # print("Predicted letter :", tokenizer.sequences_to_texts(Y_pred + 1)[0][-1])
    #
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
    #
    # print("Test1 :", complete_text("t", temperature=0.2))
    # print("Test2 :",complete_text("w", temperature=1))
    # print("Test3 :",complete_text("w", temperature=2))

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    # model = keras.models.Sequential([
    #     keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id],
    #                      dropout=0.2, recurrent_dropout=0.2),
    #     keras.layers.GRU(128, return_sequences=True,
    #                      dropout=0.2, recurrent_dropout=0.2),
    #     keras.layers.TimeDistributed(keras.layers.Dense(max_id,
    #                                                     activation="softmax"))
    # ])

    dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
    dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_length))
    dataset = dataset.batch(1)
    dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
    dataset = dataset.map(
        lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
    dataset = dataset.prefetch(1)

    # model = keras.models.Sequential([
    #     keras.layers.GRU(128, return_sequences=True, stateful=True,
    #                      dropout=0.2, recurrent_dropout=0.2,
    #                      batch_input_shape=[batch_size, None, max_id]),
    #     keras.layers.GRU(128, return_sequences=True, stateful=True,
    #                      dropout=0.2, recurrent_dropout=0.2),
    #     keras.layers.TimeDistributed(keras.layers.Dense(max_id,
    #                                                     activation="softmax"))
    # ])

    model = keras.models.Sequential([
        keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id],
                         dropout=0.2, recurrent_dropout=0.2),
        keras.layers.GRU(128, return_sequences=True,
                         dropout=0.2, recurrent_dropout=0.2),
        keras.layers.TimeDistributed(keras.layers.Dense(max_id,
                                                        activation="softmax"))
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    model.fit(dataset, epochs=20, callbacks=[ResetStatesCallback()])

    X_new = preprocess(["How are yo"])
    Y_pred = np.argmax(model.predict(X_new), axis=-1)  # DEPRECATETD model.predict_classes(X_new)
    print("V2 - Predicted letter :", tokenizer.sequences_to_texts(Y_pred + 1)[0][-1])

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


def run_test2():
    (X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data()
    print(X_train[0][:10])
    print(type(X_train), X_train.shape)

    word_index = keras.datasets.imdb.get_word_index()
    # print("word index :", word_index)
    id_to_word = {id_ + 3: word for word, id_ in word_index.items()}

    for id_, token in enumerate(("<pad>", "<sos>", "<unk>")):
        id_to_word[id_] = token

    print(" ".join([id_to_word[id_] for id_ in X_train[0][:10]]))


def run_test3():
    import tensorflow_datasets as tfds

    datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
    train_size = info.splits["train"].num_examples

    def preprocess(X_batch, y_batch):
        X_batch = tf.strings.substr(X_batch, 0, 300)
        X_batch = tf.strings.regex_replace(X_batch, b"<br\\s*/?>", b" ")
        X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
        X_batch = tf.strings.split(X_batch)
        return X_batch.to_tensor(default_value=b"<pad>"), y_batch

    from collections import Counter
    vocabulary = Counter()
    for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
        for review in X_batch:
            vocabulary.update(list(review.numpy()))

    print(vocabulary.most_common()[:3])

    vocab_size = 10000
    truncated_vocabulary = [
        word for word, count in vocabulary.most_common()[:vocab_size]]

    words = tf.constant(truncated_vocabulary)
    word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
    vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
    num_oov_buckets = 1000
    table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

    print(table.lookup(tf.constant([b"This movie was faaaaaantastic".split()])))

    def encode_words(X_batch, y_batch):
        return table.lookup(X_batch), y_batch

    train_set = datasets["train"].batch(32).map(preprocess)
    train_set = train_set.map(encode_words).prefetch(1)

    embed_size = 128
    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,
                               input_shape=[None]),
        keras.layers.GRU(128, return_sequences=True),
        keras.layers.GRU(128),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    history = model.fit(train_set, epochs=5)


def run_test4_with_mask():
    import tensorflow_datasets as tfds

    datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
    train_size = info.splits["train"].num_examples

    def preprocess(X_batch, y_batch):
        X_batch = tf.strings.substr(X_batch, 0, 300)
        X_batch = tf.strings.regex_replace(X_batch, b"<br\\s*/?>", b" ")
        X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
        X_batch = tf.strings.split(X_batch)
        return X_batch.to_tensor(default_value=b"<pad>"), y_batch

    from collections import Counter
    vocabulary = Counter()
    for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
        for review in X_batch:
            vocabulary.update(list(review.numpy()))

    print(vocabulary.most_common()[:3])

    vocab_size = 10000
    truncated_vocabulary = [
        word for word, count in vocabulary.most_common()[:vocab_size]]

    words = tf.constant(truncated_vocabulary)
    word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
    vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
    num_oov_buckets = 1000
    table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

    print(table.lookup(tf.constant([b"This movie was faaaaaantastic".split()])))

    def encode_words(X_batch, y_batch):
        return table.lookup(X_batch), y_batch

    train_set = datasets["train"].batch(32).map(preprocess)
    train_set = train_set.map(encode_words).prefetch(1)

    embed_size = 128

    # WITHOUT Mask
    # model = keras.models.Sequential([
    #     keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,
    #                            input_shape=[None]),
    #     keras.layers.GRU(128, return_sequences=True),
    #     keras.layers.GRU(128),
    #     keras.layers.Dense(1, activation="sigmoid")
    # ])

    # WITH mask
    K = keras.backend
    inputs = keras.layers.Input(shape=[None])
    mask = keras.layers.Lambda(lambda inputs: K.not_equal(inputs, 0))(inputs)
    z = keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size)(inputs)
    z = keras.layers.GRU(128, return_sequences=True)(z, mask=mask)
    z = keras.layers.GRU(128)(z, mask=mask)
    outputs = keras.layers.Dense(1, activation="sigmoid")(z)
    model = keras.Model(inputs=[inputs], outputs=[outputs])

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    # tensorboard --logdir logs
    history = model.fit(train_set, epochs=5, callbacks=[tensorboard_callback])

    plot_history_without_val(history)
    plt.show()


def test_encoder_decoder():
    pass


if __name__ == "__main__":
    check_version_proxy_gpu()
    # run_test1()
    # run_test2()
    # run_test3()
    run_test4_with_mask()
    # test_encoder_decoder(
