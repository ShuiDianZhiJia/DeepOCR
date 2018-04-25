# -*- coding:utf-8 -*-
"""
@File      : inference.py
@Software  : OCR
@Time      : 2018/3/30 17:21
@Author    : yubb
"""
import tensorflow as tf

# 48 x 48
INPUT_NODE = 2304
IMAGE_SIZE = 48
NUM_CHANNELS = 1

# 第一层卷积层
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第二层卷积层
CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 128


def inference(inputs, is_training, regularizer=None):
    # 第一层卷积层
    with tf.variable_scope('layer1-conv1', reuse=tf.AUTO_REUSE):
        weight_1 = tf.get_variable('weights', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias_1 = tf.get_variable('biases', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(inputs, weight_1, strides=[1, 1, 1, 1], padding='SAME', name='conv1')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias_1))
    # 第二层最大池化层 输出: 24x24x32
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 第三层卷积层
    with tf.variable_scope('layer3-conv2', reuse=tf.AUTO_REUSE):
        weight_2 = tf.get_variable('weights', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias_2 = tf.get_variable('biases', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, weight_2, strides=[1, 2, 2, 1], padding='SAME', name='conv2')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias_2))
    # 第四层最大池化层 输出: 12x12x64
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 将pool2转为全连接层的输入格式
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
    # 第五层全连接层
    with tf.variable_scope('layer5-fc1', reuse=tf.AUTO_REUSE):
        fc1_weight = tf.get_variable('weights', [nodes, FC_SIZE],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc1_bias = tf.get_variable('biases', [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weight) + fc1_bias)
        if regularizer:
            tf.add_to_collection('losses', regularizer(fc1_weight))
        if is_training:
            fc1 = tf.nn.dropout(fc1, keep_prob=0.5, name='drop1')
        # 批标准化
        # fc1 = tf.layers.batch_normalization(fc1, name='bn1')
    # 第六层全连接层
    with tf.variable_scope('layer6-fc2', reuse=tf.AUTO_REUSE):
        fc2_weight = tf.get_variable('weights', [FC_SIZE, NUM_CHANNELS],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_bias = tf.get_variable('biases', [FC_SIZE], initializer=tf.constant_initializer(0.1))
        if regularizer:
            tf.add_to_collection('losses', regularizer(fc2_weight))
        logit = tf.matmul(fc1, fc2_weight) + fc2_bias
    return logit


def triplet_loss(anchor, positive, negative, alpha):
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    return loss