import tensorflow as tf
import numpy as np
from tensorflow import keras

# Length of the vocabulary in chars
from senquence.text_generation.data_feed import vocab, dataset, text, char2idx, BATCH_SIZE, seq_length

vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
units = 1024
def tofloat(input,label):
    return tf.cast(input,tf.float32) ,tf.cast(label,tf.float32)



dataset = dataset.map(tofloat)
print(dataset.output_shapes)
print(dataset.output_types)
inputs = tf.keras.Input(batch_shape=[BATCH_SIZE,seq_length])  # Returns a placeholder tensor

# A layer instance is callable on a tensor, and returns a tensor.
x = keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
x = keras.layers.CuDNNGRU(units,
                                return_sequences=True,
                                recurrent_initializer='glorot_uniform',
                                stateful=True)(x)
predictions = keras.layers.Dense(vocab_size)(x)
print(predictions.dtype)
model = tf.keras.Model(inputs=inputs, outputs=predictions)

# model = keras.Sequential()
# model.add(keras.layers.Embedding(vocab_size, embedding_dim,batch_input_shape=[BATCH_SIZE,seq_length]))
#
# model.add(keras.layers.CuDNNGRU(units,
#                                 return_sequences=True,
#                                 recurrent_initializer='glorot_uniform',
#                                 stateful=True))
#
# model.add(keras.layers.Dense(vocab_size))



model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.losses.sparse_softmax_cross_entropy)
# model.summary()
# model.build(tf.TensorShape([BATCH_SIZE, seq_length]))
# hidden = model.reset_states()

# EPOCHES = 5
# dataset = dataset.repeat(EPOCHES)
#
# model.fit(dataset,epochs=EPOCHES,steps_per_epoch=272,verbose=1)
#
#
#
# start_string = 'Q'
#
# # Converting our start string to numbers (vectorizing)
# input_eval = [char2idx[s] for s in start_string]
# input_eval = tf.expand_dims(input_eval, 0)
# print(input_eval)
#
# # Empty string to store our results
# text_generated = []
# prediction = model.predict(input_eval)
# print(prediction)