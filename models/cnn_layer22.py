import tensorflow as tf
import tflearn as tfl
from tensorflow.contrib.layers import l2_regularizer
CONVSIZE1 = 3

##一层Fc层， 不给卷积层##############################

def conv_2d(inputs, filters,kernel = [3, 3], repeat_times=1,is_training=True, scope=""):
    assert repeat_times > 0
    net = inputs
    for i in range(repeat_times):
        scope_name = "{0}_{1}".format(scope, i+1)
        net = tf.layers.conv2d(net, filters, kernel,
                               activity_regularizer= l2_regularizer(0.000001),name=scope_name)
        net = tf.layers.batch_normalization(net, training=is_training, name=scope_name+"_batch")
        net = tf.nn.relu(net)
    return net

def inference(inputs,
          num_classes = 2,
          dropout_keep_prob=0.8,
          is_training=True,
          spatial_squeeze=True,
          scope='vgg_a',
          fc_conv_padding='VALID',
          global_pool=False):

        #等于下面注释的代码
        # 224x224x3 ->56x56x16
        end_points = {}
        with tf.variable_scope('layer1'):

            net = conv_2d(inputs, 16, [3, 3],repeat_times=2, is_training=is_training, scope='conv')
            net = tf.layers.max_pooling2d(net, pool_size=[4, 4], strides=4,
                                          padding='same', name='pool')  # 32
            end_point = "layer1"
            end_points[end_point] = net
        #第二层定义
        # 56x56x32 ->14x14x64
        with tf.variable_scope('layer2'):
            net = conv_2d(net, 32, [3, 3], repeat_times=2, is_training=is_training, scope='conv')
            net = tf.layers.max_pooling2d(net, pool_size=[4, 4], strides=4,
                                          padding='same', name='pool') #8
            end_point = "layer2"
            end_points[end_point] = net

        # #第三层
        # 14x14x128->3x3
        with tf.variable_scope('layer3'):
            net = conv_2d(net, 64, [3, 3], repeat_times=1,
                          is_training=is_training, scope='conv')
            net = tf.layers.max_pooling2d(net, pool_size=[4, 4], strides=4,
                                          padding='same', name='pool') #8
            print(net)
            end_point = "layer3"
            end_points[end_point] = net

        # with tf.variable_scope('layer4'):
        #     net = conv_2d(net, 64, [3, 3], repeat_times=2,
        #                   is_training=is_training, scope='conv')
        #     net = tf.layers.max_pooling2d(net, pool_size=[3, 3], strides=1,
        #                                   padding='same', name='pool') #8
        #     end_point = "layer4"
        #     end_points[end_point] = net

        with tf.variable_scope('layer5'):
            net = tf.layers.flatten(net, name='flattern')
            net = tf.layers.dropout(net, dropout_keep_prob,
                                    training=is_training, name='dropout')
            net = tf.layers.dense(net, num_classes, activation=None, name='logits')

        return  net, end_points