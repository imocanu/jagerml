import tempfile
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3)
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam')
model.summary()


def dataset_mnist():
    split = tfds.Split.TRAIN
    dataset = tfds.load('iris', split=split, as_supervised=True)
    dataset = dataset.map(lambda features, labels: ({'dense_input': features}, labels))
    dataset = dataset.batch(32).repeat()
    return dataset


for features_batch, labels_batch in dataset_mnist().take(1).cache():
    print(features_batch)
    print(labels_batch)


model_dir = tempfile.mkdtemp()
keras_estimator = tf.keras.estimator.model_to_estimator(
    keras_model=model, model_dir=model_dir)

keras_estimator.train(input_fn=dataset_mnist, steps=500)
eval_result = keras_estimator.evaluate(input_fn=dataset_mnist, steps=10)
print('Eval result: {}'.format(eval_result))
