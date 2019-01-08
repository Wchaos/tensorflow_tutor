import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#
# fashion_minist = keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_minist.load_data()
# np.savez("./data/train_data",images=train_images,labels=train_labels)
# np.savez("./data/test_data",images=test_images,labels=test_labels)

train_images = np.load("./data/train_data.npz")['images']
train_labels = np.load("./data/train_data.npz")['labels']
test_images = np.load("./data/test_data.npz")['images']
test_labels = np.load("./data/test_data.npz")['labels']

# 查看数据
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 编译模型
# AdamOptimizer，改进的随机梯度下降法，步长会动态调整
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 训练模型
model.fit(train_images, train_labels, epochs=5)

# 评估准确率
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# 做出预测
predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))

# # 保存模型
# # Save weights to a TensorFlow Checkpoint file
# model.save_weights('./weights/my_model')
#
# json_string = model.to_json()
# print(json_string)
#
# model.save('my_model.h5')