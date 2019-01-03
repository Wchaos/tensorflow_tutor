import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


train_images = np.load("./data/train_data.npz")['images']
train_labels = np.load("./data/train_data.npz")['labels']
test_images = np.load("./data/test_data.npz")['images']
test_labels = np.load("./data/test_data.npz")['labels']


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0


# reshape trian_x, test_x
#train_x = train_x.values.reshape(-1, 28, 28, 1)
#test_x = test_x.values.reshape(-1, 28, 28, 1)
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# 吧标签列转化为one-hot 编码格式
train_labels = keras.utils.to_categorical(train_labels, num_classes = 10)
#从训练数据中分出十分之一的数据作为验证数据
random_seed = 3
train_images , val_images , train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=random_seed)



model = keras.Sequential()
# 第一个卷积层，32个卷积核，大小５x5，卷积模式SAME,激活函数relu,输入张量的大小
model.add(keras.layers.Conv2D(filters= 32, kernel_size=(5,5), padding='Same', activation='relu',input_shape=(28,28,1)))
model.add(keras.layers.Conv2D(filters= 32, kernel_size=(5,5), padding='Same', activation='relu'))
# 池化层,池化核大小２x2
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
# 随机丢弃四分之一的网络连接，防止过拟合
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv2D(filters= 64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(keras.layers.Conv2D(filters= 64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(keras.layers.Dropout(0.25))
# 全连接层,展开操作，
model.add(keras.layers.Flatten())
# 添加隐藏层神经元的数量和激活函数
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.25))
# 输出层
model.add(keras.layers.Dense(10, activation='softmax'))


# 设置优化器
# lr :学习效率，　decay :lr的衰减值
optimizer = keras.optimizers.RMSprop(lr = 0.001, decay=0.0)

# 编译模型
# loss:损失函数，metrics：对应性能评估函数
model.compile(optimizer=optimizer, loss = 'categorical_crossentropy',metrics=['accuracy'])


# keras的callback类提供了可以跟踪目标值，和动态调整学习效率
# moitor : 要监测的量，这里是验证准确率
# matience: 当经过３轮的迭代，监测的目标量，仍没有变化，就会调整学习效率
# verbose : 信息展示模式，去０或１
# factor :　每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
# mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
# epsilon：阈值，用来确定是否进入检测值的“平原区”
# cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
# min_lr：学习率的下限
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor = 'val_acc', patience = 3,
                                            verbose = 1, factor=0.5, min_lr = 0.00001)

# 数据增强处理，提升模型的泛化能力，也可以有效的避免模型的过拟合
# rotation_range : 旋转的角度
# zoom_range : 随机缩放图像
# width_shift_range : 水平移动占图像宽度的比例
# height_shift_range
# horizontal_filp : 水平反转
# vertical_filp : 纵轴方向上反转
data_augment = keras.preprocessing.image.ImageDataGenerator(rotation_range= 10,zoom_range= 0.1,
                                  width_shift_range = 0.1,height_shift_range = 0.1,
                                  horizontal_flip = False, vertical_flip = False)


epochs = 40
batch_size = 100
history = model.fit_generator(data_augment.flow(train_images, train_labels, batch_size=batch_size),
                             epochs= epochs, validation_data = (val_images,val_labels),
                             verbose =2, steps_per_epoch=train_images.shape[0]//batch_size,
                             callbacks=[learning_rate_reduction])

