from dataset import *


def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    targets, contexts, labels = [], [], []

    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    for sequence in tqdm.tqdm(sequences):

        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)

        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed,
                name="negative_sampling")

            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * num_ns, dtype="int64")

            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels


def custom_loss(x_logit, y_true):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)


class Word2Vec(keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(Word2Vec, self).__init__()
        self.target_embedding = keras.layers.Embedding(vocab_size,
                                                       embedding_dim,
                                                       input_length=1,
                                                       name="w2v_embedding", )
        self.context_embedding = keras.layers.Embedding(vocab_size,
                                                        embedding_dim,
                                                        input_length=num_ns + 1)
        self.dots = keras.layers.Dot(axes=(3, 2))
        self.flatten = keras.layers.Flatten()

    def call(self, pair):
        target, context = pair
        we = self.target_embedding(target)
        ce = self.context_embedding(context)
        dots = self.dots([ce, we])
        return self.flatten(dots)


def w2v():
    IMG_SIZE = 28
    IMG_CHANNEL = 1
    BATCH_SIZE = 1024
    BUFFER_SIZE = 10000
    EPOCHS = 40
    LABELS = 1
    SEED = 42
    AUTOTUNE = tf.data.AUTOTUNE
    # for Vocabular
    VOCAB_SIZE = 4096
    SEQUENCE_LENGTH = 10

    url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
    fname = "shakespeare.txt"
    path_to_file = tf.keras.utils.get_file(fname, url)

    with open(path_to_file) as f:
        lines = f.read().splitlines()
    # for line in lines[:20]:
    #     print(line)

    text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))

    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        return tf.strings.regex_replace(lowercase,
                                        '[%s]' % re.escape(string.punctuation), '')

    vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=custom_standardization,
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=SEQUENCE_LENGTH)

    vectorize_layer.adapt(text_ds.batch(BATCH_SIZE))

    inverse_vocab = vectorize_layer.get_vocabulary()

    # print(inverse_vocab[:20])

    def vectorize_text(text):
        text = tf.expand_dims(text, -1)
        return tf.squeeze(vectorize_layer(text))

    text_vector_ds = text_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()

    sequences = list(text_vector_ds.as_numpy_iterator())
    # print(len(sequences))

    for seq in sequences[:5]:
        print(f"{seq} => {[inverse_vocab[i] for i in seq]}")

    targets, contexts, labels = generate_training_data(
        sequences=sequences,
        window_size=2,
        num_ns=4,
        vocab_size=VOCAB_SIZE,
        seed=SEED)
    print(len(targets), len(contexts), len(labels))

    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    print(dataset)

    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    print(dataset)

    embedding_dim = 128
    word2vec = Word2Vec(VOCAB_SIZE, embedding_dim, num_ns=4)
    word2vec.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

    history = word2vec.fit(dataset, epochs=EPOCHS, callbacks=[tensorboard_callback])

    plot_history_without_val(history, EPOCHS)
    plt.show()

def example():
    sentence = "The wide road shimmered in the hot sun"
    tokens = list(sentence.lower().split())
    print(tokens)


if __name__ == "__main__":
    check_version_proxy_gpu()
    # w2v()
    example()
