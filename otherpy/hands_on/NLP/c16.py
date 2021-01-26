from imports import *


def run_test1():
    check_version_proxy_gpu()
    url = "https://homl.info/shakespeare"
    fname = "shakespeare.txt"
    file_path = keras.utils.get_file(fname, url)

    with open(file_path) as f:
        shakespeare_text = f.read()

    # print(shakespeare)
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(shakespeare_text)

    print(tokenizer.texts_to_sequences(["First"]))
    max_id = len(tokenizer.word_index)
    print("[*] Number of unique words :", len(tokenizer.word_index))
    print("[*] Number of total words  :", tokenizer.document_count)

    [encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1

    train_size = tokenizer.document_count * 90 // 100
    dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

    n_steps = 100
    window_length = n_steps + 1  # target = input shifted 1 character ahead
    dataset = dataset.window(window_length, shift=1, drop_remainder=True)

    dataset = dataset.flat_map(lambda window: window.batch(window_length))

    batch_size = 32
    dataset = dataset.shuffle(10000).batch(batch_size)
    dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))

    dataset = dataset.map(
        lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))

    dataset = dataset.prefetch(1)

    model = keras.models.Sequential([
        keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id],
                         dropout=0.2, recurrent_dropout=0.2),
        keras.layers.GRU(128, return_sequences=True,
                         dropout=0.2, recurrent_dropout=0.2),
        keras.layers.TimeDistributed(keras.layers.Dense(max_id,
                                                        activation="softmax"))
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    history = model.fit(dataset, epochs=20)

    def preprocess(texts):
        X = np.array(tokenizer.texts_to_sequences(texts)) - 1
        return tf.one_hot(X, max_id)

    X_new = preprocess(["How are yo"])
    Y_pred = model.predict_classes(X_new)
    print("Predicted letter :", tokenizer.sequences_to_texts(Y_pred + 1)[0][-1])


if __name__ == "__main__":
    run_test1()
