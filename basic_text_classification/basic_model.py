import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

imdb = keras.datasets.imdb
# # 参数 num_words=10000 会保留训练数据中出现频次在前 10000 位的字词
# (train_input, train_labels), (test_input, test_labels) = imdb.load_data(num_words=10000)
# np.savez("./data/train_data",input=train_input,labels=train_labels)
# np.savez("./data/test_data",input=test_input,labels=test_labels)

train_input = np.load("./data/train_data.npz")['input']
train_labels = np.load("./data/train_data.npz")['labels']
test_input = np.load("./data/test_data.npz")['input']
test_labels = np.load("./data/test_data.npz")['labels']

# 查看数据
print("Training entries: {}, labels: {}".format(len(train_input), len(train_labels)))
print(train_input[0])
# 影评的长度可能不同
print(len(train_input[0]), len(train_input[1]))

# 建立数字到文字的映射
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()
# The first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(train_input[0]))

# 截短或者补全句子为maxlen的长度,用value填充,padding=post代表从结尾填充
train_data = keras.preprocessing.sequence.pad_sequences(train_input,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_input,
                                                       value=word_index["<PAD>"], padding='post',
                                                       maxlen=256)
print(len(train_data[0]), len(train_data[1]))

print(train_data[0])

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()

# 编译模型
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
# 创建验证集
val_input = train_input[:10000]
train_input = train_input[10000:]

val_labels = train_labels[:10000]
train_labels = train_labels[10000:]

history = model.fit(train_input,
                    train_labels,
                    epochs=40,
                    batch_size=512,
                    validation_data=(val_input, val_labels),
                    verbose=1)
results = model.evaluate(test_input, test_labels)

print(results)