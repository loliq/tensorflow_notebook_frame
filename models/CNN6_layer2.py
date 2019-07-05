import tensorflow as tf
import tflearn as tfl
import tensorflow.contrib.slim as slim
CONVSIZE1 = 3

def use_dropOut(input, dropout_keep_prob, scope):
    return slim.dropout(input, dropout_keep_prob, scope=scope)
def use_self(input):
    return  input

def inference(inputs,
          num_classes = 2,
          dropout_keep_prob=0.8,
          is_training=True,
          spatial_squeeze=True,
          scope='vgg_a',
          fc_conv_padding='VALID',
          global_pool=False):

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn = tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(0.00001),  # 越小惩罚项越重
                        normalizer_fn = tf.nn.batch_normalization):
        #等于下面注释的代码
        # 224x224x3 ->56x56x16
        end_points = {}
        with tf.variable_scope('layer1'):
            net = slim.conv2d(inputs, 16, [3, 3], scope='conv')
            net = slim.max_pool2d(net, kernel_size=[4, 4], stride=4, scope='pool')  # 32
            end_point = "layer1"
            end_points[end_point] = net
        #第二层定义
        # 56x56x32 ->14x14x64
        with tf.variable_scope('layer2'):
            net = slim.conv2d(net, 32, [3, 3], scope='conv')
            net = slim.max_pool2d(net, kernel_size=[4, 4], stride=4, scope='pool') #16
            end_point = "layer2"
            end_points[end_point] = net

        # #第三层
        # 14x14x128->7x7
        with tf.variable_scope('layer3'):
            net = slim.conv2d(net, 64, [3, 3], scope='conv')
            net = slim.max_pool2d(net, kernel_size=[4, 4], stride=4, scope='pool') #8
            end_point = "layer3"
            end_points[end_point] = net
        # # 14x14x128 -> 7x7x256
        # with tf.variable_scope('layer4'):
        #     net = slim.conv2d(net, 128, [3, 3], scope='conv')
        #     net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool')  # 8
        #     end_point = "layer4"
        #     end_points[end_point] = net
        with tf.variable_scope('layer6'):
            net = slim.conv2d(net, 256, [3, 3], scope='conv')
            net = slim.avg_pool2d(net, kernel_size=[3, 3], stride=1, scope='pool')
            end_point = "layer6"
            end_points[end_point] = net

        with tf.variable_scope('layer7'):
            net = slim.flatten(net, scope='flattern')
            net = slim.fully_connected(net, 128, scope='fc6')
            end_point = "layer7"
            end_points[end_point] = net

        with tf.variable_scope('layer8'):
            net = slim.fully_connected(net, 64, scope='fc6')
            end_point = "layer8"
            end_points[end_point] = net


        with tf.variable_scope('layer9'):
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout')
            net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc8')

    return  net, end_points