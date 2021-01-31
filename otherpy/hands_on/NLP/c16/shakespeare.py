from to_import import *


def run_test():
    url = "https://homl.info/shakespeare"
    filepath = keras.utils.get_file("shakespeare.txt", url)
    with open(filepath) as f:
        shakespeare_text = f.read()

    print(type(shakespeare_text))
    print(len(shakespeare_text))
    split_lines = shakespeare_text.splitlines()
    print(len(split_lines))
    print(type(split_lines))
    print(split_lines[:10])

    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(shakespeare_text)

    MAX_ID = len(tokenizer.word_index)
    DATASET_SIZE = tokenizer.document_count
    print(MAX_ID, DATASET_SIZE)

    [encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
    # print(len(encoded))

    TRAIN_SIZE = DATASET_SIZE * 1 // 100
    # print(TRAIN_SIZE)

    dataset = tf.data.Dataset.from_tensor_slices(encoded[:TRAIN_SIZE])
    # for i in dataset.take(3):
    #    print(i)

    N_STEPS = 100
    WINDOW_LENGTH = N_STEPS + 1

    dataset = dataset.window(WINDOW_LENGTH, shift=1, drop_remainder=True)
    # for i in dataset.take(3):
    #    print(i)

    dataset = dataset.flat_map(lambda window: window.batch(WINDOW_LENGTH))
    # for i in dataset.take(3):
    #    print(i)

    BATCH_SIZE = 32
    dataset = dataset.shuffle(10000).batch(BATCH_SIZE)

    dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
    # for x, y in dataset.take(1):
    #    print(x)
    #    print(y)

    dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=MAX_ID), Y_batch))
    # for x, y in dataset.take(1):
    #    print(x)
    #    print(y)

    dataset = dataset.prefetch(1)
    EPOCHS = 10
    model = keras.models.Sequential([
        keras.layers.GRU(128, return_sequences=True, input_shape=[None, MAX_ID],
                         dropout=0.2,
                         recurrent_dropout=0.2),
        keras.layers.GRU(128, return_sequences=True,
                         dropout=0.2,
                         recurrent_dropout=0.2),
        keras.layers.TimeDistributed(keras.layers.Dense(MAX_ID, activation="softmax"))
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    history = model.fit(dataset, epochs=EPOCHS)

    model.save('model.h5')
    # model.evaluate(test_images, test_labels, verbose=2)

    plot_history_per_keys(history, epochs=EPOCHS)
    plt.show()

    def preprocess(texts):
        X = np.array(tokenizer.texts_to_sequences(texts)) - 1
        return tf.one_hot(X, MAX_ID)

    X_new = preprocess(["How are yo"])
    Y_pred = np.argmax(model.predict(X_new), axis=-1)
    print("Predicted letter :", tokenizer.sequences_to_texts(Y_pred + 1)[0][-1])


def load_saved_model(path="model.h5"):
    url = "https://homl.info/shakespeare"
    filepath = keras.utils.get_file("shakespeare.txt", url)
    with open(filepath) as f:
        shakespeare_text = f.read()

    print(type(shakespeare_text))
    print(len(shakespeare_text))
    split_lines = shakespeare_text.splitlines()
    print(len(split_lines))
    print(type(split_lines))
    print(split_lines[:10])

    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(shakespeare_text)

    MAX_ID = len(tokenizer.word_index)

    model = tf.keras.models.load_model(path)
    model.summary()

    def preprocess(texts):
        X = np.array(tokenizer.texts_to_sequences(texts)) - 1
        return tf.one_hot(X, MAX_ID)

    def next_char(text, temperature=1):
        X_new = preprocess([text])
        y_proba = model.predict(X_new)[0, -1:, :]
        rescaled_logits = tf.math.log(y_proba) / temperature
        char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
        return tokenizer.sequences_to_texts(char_id.numpy())[0]

    def complete_text(text, n_chars=50, temperature=1):
        for _ in range(n_chars):
            text += next_char(text, temperature)
        return text

    # TEST
    print("[test] :",complete_text("t", temperature=0.2))
    print("[test] :",complete_text("h", temperature=1))
    print("[test] :",complete_text("i", temperature=2))


if __name__ == "__main__":
    check_version_proxy_gpu()
    # run_test()
    load_saved_model()
