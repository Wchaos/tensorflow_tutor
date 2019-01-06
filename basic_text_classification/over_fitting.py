import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

train_input = np.load("./data/train_data.npz")['input']
train_labels = np.load("./data/train_data.npz")['labels']
test_input = np.load("./data/test_data.npz")['input']
test_labels = np.load("./data/test_data.npz")['labels']

NUM_WORDS = 10000


def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
    return results


train_input = multi_hot_sequences(train_input, dimension=NUM_WORDS)
test_input = multi_hot_sequences(test_input, dimension=NUM_WORDS)

plt.plot(train_input[0])
plt.show()


def build_baseline_model():
    model = keras.Sequential([
        # `input_shape` is only required here so that `.summary` works.
        keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy', 'binary_crossentropy'])

    model.summary()
    return model



baseline_model = build_baseline_model()
baseline_history = baseline_model.fit(train_input,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_input,test_labels),
                                      verbose=2)


def build_smaller_model():
    pass
