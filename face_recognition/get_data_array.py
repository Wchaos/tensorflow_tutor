from PIL import Image
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

names = ["Abhishek_Bachan", "Alex_Rodriguez", "Ali_Landry", "Alyssa_Milano", "Anderson_Cooper", "Anna_Paquin",
         "Audrey_Tautou", "Barack_Obama", "Ben_Stiller", "Christina_Ricci", "Clive_Owen", "Cristiano_Ronaldo",
         "Daniel_Craig", "Danny_Devito", "David_Duchovny", "Denise_Richards", "Diane_Sawyer", "Donald_Faison",
         "Ehud_Olmert", "Faith_Hill", "Famke_Janssen", "Hugh_Jackman", "Hugh_Laurie", "James_Spader", "Jared_Leto",
         "Julia_Roberts",
         "Julia_Stiles", "Karl_Rove", "Katherine_Heigl", "Mark_Ruffalo", "Meg_Ryan", "Michelle_Trachtenberg",
         "Michelle_Wie", "Mickey_Rourke", "Miley_Cyrus", "Milla_Jovovich", "Nicole_Richie", "Rachael_Ray",
         "Robert_Gates", "Ryan_Seacrest", "Sania_Mirza", "Sarah_Chalke", "Sarah_Palin", "Scarlett_Johansson",
         ]
dir_train = "./data/face_train.tfrecords"  # face data sets 11 persons  11个人的人脸信息

dir_test = "./data/face_test.tfrecords"


def spars2narry(name):
    image = Image.open(name)
    img = np.array(image)
    return img


def showimg(img):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()


def get_data():
    dir_ = 'D:\\dpl\\handwriting_recognition\\handwriting_recognition\\face_image_rect2'
    test_dir = dir_ + '\\test'
    train_dir = dir_ + '\\train'
    train_files = os.listdir(train_dir)
    l1 = len(train_files)
    test_files = os.listdir(test_dir)
    l2 = len(test_files)
    train_data = np.empty([l1, 1, 128, 128], dtype=int)
    test_data = np.empty([l1, 1, 128, 128], dtype=int)
    train_lb = np.empty([l1, 11], dtype=int)
    test_lb = np.empty([l1, 11], dtype=int)
    for i in range(l1):
        path = train_dir + '\\' + train_files[i]
        img = spars2narry(path)
        # if i%500==0:
        #
        #     print(img)
        #     showimg(img)
        train_data[i][0] = img
        tt = train_files[i].split("_")
        name = tt[0] + "_" + tt[1]
        idx = names.index(name)
        one_hot = np.zeros([11])
        one_hot[idx] = 1
        train_lb[i] = one_hot

    for i in range(l2):
        path = test_dir + '\\' + test_files[i]
        img = spars2narry(path)
        test_data[i][0] = img
        # if i%500==0:
        #     print(img)
        #     showimg(img)
        tt = test_files[i].split("_")
        name = tt[0] + "_" + tt[1]
        idx = names.index(name)
        one_hot = np.zeros([11])
        one_hot[idx] = 1
        test_lb[i] = one_hot

    return (train_data, train_lb), (test_data, test_lb)


def get_data2():
    with tf.Session() as sess:
        (train_images, train_labels), (test_images, test_labels) = prepare_data()
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 启动队列
        bx, batch_ys = sess.run([train_images, train_labels])  # 取出数据
        tex, tey = sess.run([test_images, test_labels])  # 取出测试数据
        coord.request_stop()
        coord.join(threads)  # 关闭队列
    # print(bx.shape)
    bbx = np.empty([2900, 128, 128, 1])
    ttx = np.empty([300, 128, 128, 1])
    bbx[:, :, :, 0] = np.reshape(bx/255.0, (2900, 128, 128))
    ttx[:, :, :, 0] = np.reshape(tex/255.0, (300, 128, 128))
    showimg(bbx[0, :, :, 0])
    print(tey[1:10])
    print(batch_ys[1:10])
    return (bbx, batch_ys), (ttx, tey)


# 从 tfrecord 文件中解析结构化数据 （特征）
def parse_image_example(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),  # image data
            'pixels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),  # number label
        })
    return features


def prepare_data():
    reader = tf.TFRecordReader()  # reader for TFRecord file
    if os.path.exists(dir_train) and os.path.exists(dir_test):
        train_queue = tf.train.string_input_producer([dir_train])
        test_queue = tf.train.string_input_producer([dir_test])  # files read queue
    else:
        raise Exception("%s or %s file doesn't exist" % (dir_train, dir_test))
    _, serialized_example = reader.read(train_queue)  # examples in TFRecord file
    _, serialized_test = reader.read(test_queue)
    features = parse_image_example(serialized_example)  # parsing features
    features_test = parse_image_example(serialized_test)
    pixels = tf.cast(features['pixels'], tf.int32)
    image = tf.decode_raw(features['image_raw'], tf.uint8)  # decode image data from string to image, a Tensor
    image.set_shape([16384])  # pixels is 16384
    label = tf.cast(features['label'], tf.int32)
    image_test = tf.decode_raw(features_test['image_raw'], tf.uint8)
    image_test.set_shape([16384])
    label_test = tf.cast(features_test['label'], tf.int32)
    # return (image, label), (image_test, label_test)
    image_batch, lb = tf.train.shuffle_batch(
        [image, label], batch_size=2900, capacity=10000,
        min_after_dequeue=500)  # queue of image_batch, shuffle_batch mean random
    label_batch = tf.one_hot(lb, 11)  # one_hot, 2 for [0,0,1,0,0,0,...]
    image_batch_test, lb_test = tf.train.shuffle_batch(
        [image_test, label_test], batch_size=300, capacity=10000, min_after_dequeue=500)
    label_batch_test = tf.one_hot(lb_test, 11)
    return (image_batch, label_batch), (image_batch_test, label_batch_test)
