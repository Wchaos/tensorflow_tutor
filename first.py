import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100))  # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300  # 点乘，设置输出的标注数据

# 构造一个线性模型
#
b = tf.Variable(tf.zeros([1]))  # 定义b为[0]
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))  # 定义权值W为1*2的矩阵
y = tf.matmul(W, x_data) + b  # 矩阵乘法

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))  # 定义损失函数，对矩阵的每个元素求平方，然后对所有元素求均值
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 定义梯度下降优化器，并设置学习率为0.5
train = optimizer.minimize(loss)  # 使用梯度下降优化器，优化损失函数

# 初始化变量
init = tf.global_variables_initializer()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))


# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]