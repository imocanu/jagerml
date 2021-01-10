from jagerml.helper import *

import io
import os
import re
import shutil
import string
import tensorflow as tf

from datetime import datetime
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

if os.path.exists("./aclImdb"):
    print("Dataset already exists ...")
    train_dir = os.path.join("./aclImdb", 'train')
    print(os.listdir(train_dir))
else:
    print("Dataset is downloading ...")
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url, untar=True, cache_dir='.', cache_subdir='')
    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    print("dataset {} dir {}".format(dataset, dataset_dir))
    print("listdir :", os.listdir(dataset_dir))
    if os.path.exists("./aclImdb/train/unsup"):
        remove_dir = os.path.join("./aclImdb/train", 'unsup')
        shutil.rmtree(remove_dir)

batch_size = 1024
seed = 123
train_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/train',
                                                              batch_size=batch_size,
                                                              validation_split=0.2,
                                                              subset='training',
                                                              seed=seed)
val_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/train',
                                                            batch_size=batch_size,
                                                            validation_split=0.2,
                                                            subset='validation',
                                                            seed=seed)

for text_batch, label_batch in train_ds.take(1):
    for i in range(1):
        print("****", label_batch[i].numpy(), text_batch.numpy()[i])

embeddingLayer = tf.keras.layers.Embedding(1000, 5)
result = embeddingLayer(tf.constant([1, 2, 3]))
print(result.numpy())


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation), '')


# Vocabulary size and number of words in a sequence.
vocab_size = 10000
sequence_length = 100

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Note that the layer uses the custom standardization defined above.
# Set maximum_sequence length as all samples are not of the same length.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and call adapt to build the vocabulary.
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

embedding_dim = 16

model = Sequential([
    vectorize_layer,
    Embedding(vocab_size, embedding_dim, name="embedding"),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1)
])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[tensorboard_callback])
