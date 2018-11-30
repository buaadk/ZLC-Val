# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 14:59:19 2018

@author: zxb
"""
#softmax

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

x = tf.placeholder(tf.float32,[None,784])
#初始化权重，偏置为0
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#predict
#y = tf.nn.softmax(tf.matmul(x,W)+b)
y = tf.nn.softmax(tf.add(tf.matmul(x,W),b))
#label
y_=tf.placeholder(tf.float32,[None,10])
#交差熵损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
#定义梯度下降优化器，学习率0.01，最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#指定默认session
sess = tf.InteractiveSession()
#初始化全部变量
tf.global_variables_initializer().run()
#迭代1000轮
for _ in range(1000):
    #随机从训练集中抽取100个样本
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
#tf.argmax() 找Tensor中最大值编号
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#tf.cast()强制类型转换，tf.reduce_mean求平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))


