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
print(train_labels[:20])
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
train_input = keras.preprocessing.sequence.pad_sequences(train_input,
                                                         value=word_index["<PAD>"],
                                                         padding='post',
                                                         maxlen=256)

test_input = keras.preprocessing.sequence.pad_sequences(test_input,
                                                        value=word_index["<PAD>"], padding='post',
                                                        maxlen=256)
print(train_input.shape, test_input.shape)

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

print(val_input.shape, val_labels.shape)
# 训练模型
history = model.fit(train_input,
                    train_labels,
                    epochs=40,
                    batch_size=512,
                    validation_data=(val_input, val_labels),
                    verbose=1)
# 测试模型
results = model.evaluate(test_input, test_labels)

print("[loss, accuracy] in Test: %s" % results)


# 创建准确率和损失随时间变化的图
# model.fit() 返回一个 History 对象，该对象包含一个字典，其中包括训练期间发生的所有情况

history_dict = history.history
print(history_dict.keys())

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()