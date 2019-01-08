import tensorflow as tf
tf.enable_eager_execution()



from senquence.text_generation.data_feed import vocab, BATCH_SIZE, seq_length, char2idx, dataset, idx2char



import numpy as np
import os
import time

class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(Model, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        if tf.test.is_gpu_available():
            self.gru = tf.keras.layers.CuDNNGRU(self.units,
                                                return_sequences=True,
                                                recurrent_initializer='glorot_uniform',
                                                stateful=True)
        else:
            self.gru = tf.keras.layers.GRU(self.units,
                                           return_sequences=True,
                                           recurrent_activation='sigmoid',
                                           recurrent_initializer='glorot_uniform',
                                           stateful=True)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, **kwargs):
        embedding = self.embedding(x)
        # output at every time step
        # output shape == (batch_size, seq_length, hidden_size)
        output = self.gru(embedding)

        # The dense layer will output predictions for every time_steps(seq_length)
        # output shape after the dense layer == (seq_length * batch_size, vocab_size)
        prediction = self.fc(output)
        # states will be used to pass at every step to the model while training
        return prediction


# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
units = 1024

model = Model(vocab_size, embedding_dim, units)

# Using adam optimizer with default arguments
optimizer = tf.train.AdamOptimizer()

# model.compile(optimizer=tf.train.AdamOptimizer(),
#               loss=tf.losses.sparse_softmax_cross_entropy)
# model.build(tf.TensorShape([BATCH_SIZE, seq_length]))
# model.summary()
#
# EPOCHES = 2
# dataset = dataset.repeat(EPOCHES)
# model.fit(dataset,epochs=EPOCHES,steps_per_epoch=172,verbose=1)

# # Using sparse_softmax_cross_entropy so that we don't have to create one-hot vectors
# def loss_function(real, preds):
#     return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)
#
#
# model.build(tf.TensorShape([BATCH_SIZE, seq_length]))
# model.summary()
#
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# print(checkpoint_prefix)
#
#
# def train_model(EPOCHS=5):
#     # Training loop
#     for epoch in range(EPOCHS):
#         start = time.time()
#
#         # initializing the hidden state at the start of every epoch
#         # initally hidden is None
#         hidden = model.reset_states()
#
#         for (batch, (inp, target)) in enumerate(dataset):
#             with tf.GradientTape() as tape:
#                 # feeding the hidden state back into the model
#                 # This is the interesting step
#                 predictions = model(inp)
#                 loss = loss_function(target, predictions)
#
#             grads = tape.gradient(loss, model.variables)
#             optimizer.apply_gradients(zip(grads, model.variables))
#
#             if batch % 100 == 0:
#                 print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
#                                                              batch,
#                                                              loss))
#         # saving (checkpoint) the model every 5 epochs
#         if (epoch + 1) % 5 == 0:
#             model.save_weights(checkpoint_prefix)
#
#         print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
#         print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
#
#     model.save_weights(checkpoint_prefix)
#
#
# # 训练模型
# train_model()
#
# print(tf.train.latest_checkpoint(checkpoint_dir))

model = Model(vocab_size, embedding_dim, units)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

# Evaluation step (generating text using the learned model)

# Number of characters to generate
num_generate = 1000

# You can change the start string to experiment
start_string = 'Q'

# Converting our start string to numbers (vectorizing)
input_eval = [char2idx[s] for s in start_string]
input_eval = tf.expand_dims(input_eval, 0)
print(input_eval)

# Empty string to store our results
text_generated = []

# Low temperatures results in more predictable text.
# Higher temperatures results in more surprising text.
# Experiment to find the best setting.
temperature = 1.0

# Evaluation loop.

# Here batch size == 1
model.reset_states()
for i in range(num_generate):
    predictions = model(input_eval)
    print(predictions)
    # remove the batch dimension
    predictions = tf.squeeze(predictions, 0)
    print(predictions)

    # using a multinomial distribution to predict the word returned by the model
    predictions = predictions / temperature
    predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()

    # We pass the predicted word as the next input to the model
    # along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id], 0)
    text_generated.append(idx2char[predicted_id])

print(start_string + ''.join(text_generated))
