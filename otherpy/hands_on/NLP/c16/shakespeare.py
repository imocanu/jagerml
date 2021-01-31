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
    print(len(encoded))

    TRAIN_SIZE = DATASET_SIZE * 10 // 100
    print(TRAIN_SIZE)

    dataset = tf.data.Dataset.from_tensor_slices(encoded[:TRAIN_SIZE])
    for i in dataset.take(3):
        print(i)

    N_STEPS = 100
    WINDOW_LENGTH = N_STEPS + 1

    dataset = dataset.window(WINDOW_LENGTH, shift=1, drop_remainder=True)
    for i in dataset.take(3):
        print(i)

    dataset = dataset.flat_map(lambda window: window.batch(WINDOW_LENGTH))
    for i in dataset.take(3):
        print(i)

    BATCH_SIZE = 32
    dataset = dataset.shuffle(10000).batch(BATCH_SIZE)

    dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
    for x, y in dataset.take(1):
        print(x)
        print(y)

    dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=MAX_ID), Y_batch))
    for x, y in dataset.take(1):
        print(x)
        print(y)

    dataset = dataset.prefetch(1)

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
    history = model.fit(dataset, epochs=20)

    def preprocess(texts):
        X = np.array(tokenizer.texts_to_sequences(texts)) - 1
        return tf.one_hot(X, MAX_ID)

    X_new = preprocess(["How are yo"])
    Y_pred = np.argmax(model.predict(X_new), axis=-1)
    print("Predicted letter :", tokenizer.sequences_to_texts(Y_pred + 1)[0][-1])





if __name__ == "__main__":
    check_version_proxy_gpu()
    run_test()
