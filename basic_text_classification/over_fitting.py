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

# plt.plot(train_input[0])
# plt.show()


def build_model(hidden_layers=16):
    model = keras.Sequential([
        # `input_shape` is only required here so that `.summary` works.
        keras.layers.Dense(hidden_layers, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        keras.layers.Dense(hidden_layers, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy', 'binary_crossentropy'])

    model.summary()
    return model



baseline_model = build_model()
baseline_history = baseline_model.fit(train_input,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_input,test_labels),
                                      verbose=2)

# smaller_model = build_model(hidden_layers=4)
# smaller_history = smaller_model.fit(train_input,
#                                     train_labels,
#                                     epochs=20,
#                                     batch_size=512,
#                                     validation_data=(test_input, test_labels),
#                                     verbose=2)
#
# bigger_model = build_model(hidden_layers=512)
# bigger_history = bigger_model.fit(train_input, train_labels,
#                                   epochs=20,
#                                   batch_size=512,
#                                   validation_data=(test_input, test_labels),
#                                   verbose=2)

def plot_history(histories, key='binary_crossentropy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])
  plt.show()

# plot_history([('baseline', baseline_history),
#               ('smaller', smaller_history),
#               ('bigger', bigger_history)])


def build_l2_model(hidden_layers=16):
    model = keras.models.Sequential([
        keras.layers.Dense(hidden_layers, kernel_regularizer=keras.regularizers.l2(0.001),
                           activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        keras.layers.Dense(hidden_layers, kernel_regularizer=keras.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy', 'binary_crossentropy'])
    model.summary()
    return model


l2_model = build_l2_model()
l2_model_history = l2_model.fit(train_input, train_labels,
                                epochs=20,
                                batch_size=512,
                                validation_data=(test_input, test_labels),
                                verbose=2)

plot_history([('baseline', baseline_history),
              ('l2', l2_model_history)])


def build_dpt_model(hidden_layers=16):
    model = keras.models.Sequential([
        keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])

    model.summary()
    return model
dpt_model = build_dpt_model()
dpt_model_history = dpt_model.fit(train_input, train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_input, test_labels),
                                      verbose=2)
plot_history([('baseline', baseline_history),
              ('dropout', dpt_model_history)])