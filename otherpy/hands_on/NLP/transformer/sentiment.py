from sklearn.datasets.tests.test_samples_generator import test_make_classification_weights_array_or_list_ok
from to_import import *
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
stop_words = stopwords.words('english')
nltk.download('punkt')


def run_sentiment():
    """
    https://levelup.gitconnected.com/natural-language-processing-and-sentiment-analysis-using-tensorflow-c2948f2623f
    """
    file_name = "https://github.com/farazkhanfk7/TensorFlow-Keras/blob/master/NaturalLanguageProcessing/20191226-reviews.csv"

    df = pd.read_csv("reviews_ds.csv")
    # df.tail(10)
    # print(df)

    def rate(ratex):
        if ratex > 3:
            return "positive"
        elif ratex == 3:
            return "neutral"
        else:
            return "negative"

    # print(df.keys())
    # for i in df["body"]:
    #     print("=>", i)
    def text_process(x):
        filtered_list = []
        text_tokens = word_tokenize(str(x))
        for w in text_tokens:
            if w not in stop_words:
                filtered_list.append(w)

        filtered_sentence = " ".join(filtered_list)
        filtered_sentence = re.sub(r'[^\w]]', '', filtered_sentence)
        filtered_sentence = re.sub(r"\d", "", filtered_sentence)
        return filtered_sentence

    df["review"] = df["title"] + df["body"]
    df["sentiment"] = df["rating"].apply(rate)
    df = df[["review", "sentiment"]]
    df["review"] = df["review"].apply(text_process)

    y = df["sentiment"]
    X = df["review"]

    X_train, \
    X_test, \
    y_train, \
    y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM)

    print(len(X_train), len(X_test), len(y_train), len(y_test))

    print(df.head())

    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    NUM_WORDS = VOCAB_SIZE = len(X_train)
    MAX_LENGTH = 40
    EMBEDDING_DIM = 16

    tk = Tokenizer(num_words=NUM_WORDS, oov_token="<oov>")
    tk.fit_on_texts(X_train)
    tk.fit_on_texts(X_test)

    X_train_seq = tk.texts_to_sequences(X_train)
    X_test_seq = tk.texts_to_sequences(X_test)

    X_train_padded = pad_sequences(X_train_seq, padding="post", maxlen=MAX_LENGTH)
    X_test_padded = pad_sequences(X_test_seq, padding="post", maxlen=MAX_LENGTH)

    y_tk = Tokenizer()
    y_tk.fit_on_texts(y_train)
    y_tk.fit_on_texts(y_test)

    y_train_seq = np.array(y_tk.texts_to_sequences(y_train))
    y_test_seq = np.array(y_tk.texts_to_sequences(y_test))

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(NUM_WORDS, EMBEDDING_DIM, input_length=MAX_LENGTH))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(6, activation="softmax"))

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    model.summary()
    EPOCHS = 20
    BATCH_SIZE = 32
    history = model.fit(X_train_padded, y_train_seq,
                        epochs=EPOCHS,
                        validation_data=(X_test_padded, y_test_seq),
                        batch_size=BATCH_SIZE)

    plot_history(history)
    plt.show()

    sentence = "love the phone"
    print("Input :", sentence)
    input_seq = tk.texts_to_sequences([sentence])
    print("Seq :", input_seq)
    input_seq_padded = pad_sequences(input_seq)
    print("Padd :", input_seq_padded)
    predicted_token = np.argmax(model.predict(input_seq_padded))
    print("[*] Predicted token :", predicted_token)


if __name__ == "__main__":
    check_version_proxy_gpu()
    run_sentiment()
