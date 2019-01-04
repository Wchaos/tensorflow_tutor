import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from face_recognition.get_data_array import get_data2

(train_images, train_labels), (test_images, test_labels) = get_data2()
print(train_images.shape)
print(train_images[0])

# print(train_images.shape)
# print(train_labels.shape)
# print(test_images.shape)
# print(test_labels.shape)

# train_images = tf.reshape(train_images, [128, 128])

# model = tf.keras.Sequential([
#     layers.Conv2D(64, (2, 2), activation='relu', padding='same', input_shape=(128, 128, 1)),
#     layers.MaxPool2D(pool_size=(2, 2), padding='same'),
#     layers.Dropout(0.8)
# ])

inputs = tf.keras.Input(shape=(128, 128, 1))  # Returns a placeholder tensor
# A layer instance is callable on a tensor, and returns a tensor.
# x = layers.Flatten(input_shape=(128, 128))(inputs)
x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
# x = layers.Dropout(0.25)(x)
print(x)
x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
# x = layers.Dropout(0.25)(x)
print(x)

x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
# x = layers.Dropout(0.25)(x)
print(x)

x = keras.layers.Flatten()(x)

y = keras.layers.Dense(625, activation='relu')(x)
y = keras.layers.Dropout(0.2)(y)
predictions = keras.layers.Dense(11, activation='softmax')(y)
print(predictions.shape)
print(predictions)
model = tf.keras.Model(inputs=inputs, outputs=predictions)
# sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])
model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels)
model.save('my_model.h5')

print('Test accuracy:', test_acc)
predictions = model.predict(test_images)
print(predictions[100:110])
