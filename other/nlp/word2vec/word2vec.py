import io
import itertools
import numpy as np
import os
import re
import string
import tensorflow as tf
import tqdm

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Dot, Embedding, Flatten, GlobalAveragePooling1D, Reshape
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

SEED = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE

sentence = "The wide road shimmerend in the hot sun"
tokens = list(sentence.lower().split())
print(len(tokens))

path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                       'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

with open(path_to_file) as f:
    lines = f.read().splitlines()
# for line in lines[:20]:
#    print(line)
text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))


def customStandardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')


vocabSize = 4096
sequenceLength = 10

vectorizeLayer = TextVectorization(
    standardize=customStandardization,
    max_tokens=vocabSize,
    output_mode="int",
    output_sequence_length=sequenceLength
)

vectorizeLayer.adapt(text_ds.batch(1024))

inverseVocab = vectorizeLayer.get_vocabulary()


# print(inverseVocab)

def vectorizeText(text):
    text = tf.expand_dims(text, -1)
    return tf.squeeze(vectorizeLayer(text))


textVectorDS = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorizeLayer).unbatch()

sequences = list(textVectorDS.as_numpy_iterator())
print(len(sequences))

for seq in sequences[:5]:
    print(f"{seq} => {[inverseVocab[i] for i in seq]}")


def generate_training_data(sequencesT, window_size, num_ns, vocab_size, seed):
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for vocab_size tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in dataset.
    for sequence in tqdm.tqdm(sequencesT):

        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)

        # Iterate over each positive skip-gram pair to produce training examples
        # with positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=SEED,
                name="negative_sampling")

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels


targets, contexts, labels = generate_training_data(
    sequencesT=sequences,
    window_size=2,
    num_ns=4,
    vocab_size=vocabSize,
    seed=SEED
)

BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
print(dataset)


class Word2Vec(Model):
    def __init__(self, vocab_size, embedding_dim_w2v):
        super(Word2Vec, self).__init__()
        self.num_ns = 4
        self.target_embedding = Embedding(vocab_size,
                                          embedding_dim_w2v,
                                          input_length=1,
                                          name="w2v_embedding", )
        self.context_embedding = Embedding(vocab_size,
                                           embedding_dim_w2v,
                                           input_length=self.num_ns + 1)
        self.dots = Dot(axes=(3, 2))
        self.flatten = Flatten()

    def call(self, pair):
        target, context = pair
        we = self.target_embedding(target)
        ce = self.context_embedding(context)
        dots = self.dots([ce, we])
        return self.flatten(dots)


def custom_loss(x_logit, y_true):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)


embedding_dim = 128
word2vec = Word2Vec(vocabSize, embedding_dim)
word2vec.compile(optimizer=tf.keras.optimizers.Adam(),
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

word2vec.fit(dataset, epochs=50, callbacks=[tensorboard_callback])