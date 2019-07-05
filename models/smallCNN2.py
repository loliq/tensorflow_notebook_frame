"""
     -*- coding: utf-8 -*-
    @Project: PyCharm
    @File    : smallCNN.py
    @Author  : LLL
    @Site    :
    @Email   : lilanluo@stu.xmu.edu.cn
    @Date    : 2019/5/7 10:28
    @info   : 搭建小型的CNN网络
"""
# 不能用slim.repeat######################################
import tensorflow as tf
import tensorflow.contrib.slim as slim

def inference(inputs, num_classes=2, is_training=False, dropout_keep_prob=0.8, reuse=tf.AUTO_REUSE,
              scope='cnn_4_airbag_3'):
    ### 1: Convolution + MaxPooling
    # 224 X 224 X 1 --> 224 X 224 X 16 --> 56 X 56 X 16
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(0.00001),  # 越小惩罚项越重
                        normalizer_fn=slim.batch_norm):
        end_points = {}
        with tf.variable_scope('layer1'):
            net = slim.conv2d(inputs, 16, [3, 3], scope='conv')
            net = slim.max_pool2d(net, [4, 4], stride=4, scope='pool')
            end_point = "layer1"
            end_points[end_point] = net

        ### 2: Convolution + MaxPooling
        # 56 ->14
        with tf.variable_scope('layer2'):
            net = slim.conv2d(net, 32, [3, 3], scope='conv')
            net = slim.max_pool2d(net, [4, 4], stride=4, scope='pool')
            end_point = "layer2"
            end_points[end_point] = net
        ### 14 -> 3
        with tf.variable_scope('layer3'):
            net = slim.conv2d(net, 128, [3, 3], scope='conv')
            net = slim.max_pool2d(net, [4, 4], stride=4, scope='pool')
            end_point = "layer3"
            end_points[end_point] = net

        with tf.variable_scope('layer4'):
            net = slim.conv2d(net, 256, [3, 3], scope='conv')
            net = slim.avg_pool2d(net, [3, 3], scope='pool')
            end_point = "layer4"
            end_points[end_point] = net

        ### 5: Full Connection + Dropout
        # 2 X 2 X 256 --> 2048 --> 64
        with tf.variable_scope('layer5'):
            net = slim.flatten(net, scope='flatten')
            net = slim.fully_connected(net, 128, scope='fc')

        with tf.variable_scope('layer6'):
            net = slim.fully_connected(net, 64, scope='fc')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout')

        ### 6: Full Connection
        # 64 --> num_classes
        with tf.variable_scope('layer7'):
            net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc')

    return net,end_points