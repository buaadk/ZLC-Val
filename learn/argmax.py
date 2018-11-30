# -*- coding: utf-8 -*-

import tensorflow as tf
# row,col  max value

A =[[1,2,3,4]]

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

with tf.Session() as sess:
    print(tf.argmax(A,0).eval())
    print(tf.argmax(A,1).eval())

# 0: col max  index num is 0
# 1: row max is num 4 index is 3