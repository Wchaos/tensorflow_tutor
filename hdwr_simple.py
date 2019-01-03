import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

sess = tf.InteractiveSession()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
ones = tf.Variable(tf.ones([10]))

sess.run(tf.global_variables_initializer())
# 定义输出的计算图
y = tf.nn.softmax(tf.matmul(x, w) + b)
# 定义损失函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y)+(ones-y_)*tf.log(ones-y))
# 定义优化算法(反向传播)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 定义模型评估函数
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 训练网络
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    if i % 100 == 0:
        print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
