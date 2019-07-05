import tensorflow as tf
import tflearn as tfl
import tensorflow.contrib.slim as slim
CONVSIZE1 = 3

##一层Fc层， 不给卷积层##############################

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
                        weights_regularizer=slim.l2_regularizer(0.000001),  # 越小惩罚项越重
                        normalizer_fn = slim.batch_norm
                        ):
        #等于下面注释的代码
        # 224x224x3 ->56x56x16
        end_points = {}
        with tf.variable_scope('layer1'):
            net = slim.repeat(inputs, 2, slim.conv2d, 16, [CONVSIZE1, CONVSIZE1], scope='conv')
            net = slim.max_pool2d(net, kernel_size=[4, 4], stride=4, scope='pool')  # 32
            end_point = "layer1"
            end_points[end_point] = net
        #第二层定义
        # 56x56x32 ->14x14x64
        with tf.variable_scope('layer2'):
            net = slim.repeat(net, 2, slim.conv2d, 32, [CONVSIZE1, CONVSIZE1], scope='conv')
            net = slim.max_pool2d(net, kernel_size=[4, 4], stride=4, scope='pool') #16
            end_point = "layer2"
            end_points[end_point] = net

        # #第三层
        # 14x14x128->3x3
        with tf.variable_scope('layer3'):
            net = slim.repeat(net, 2,slim.conv2d, 64, [CONVSIZE1, CONVSIZE1], scope='conv')
            net = slim.max_pool2d(net, kernel_size=[4, 4], stride=4, scope='pool') #8
            end_point = "layer3"
            end_points[end_point] = net

        with tf.variable_scope('layer4'):
            net = slim.repeat(net, 2, slim.conv2d, 128, [CONVSIZE1, CONVSIZE1], scope='conv')
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=1, scope='pool')
            end_point = "layer3"
            end_points[end_point] = net

        with tf.variable_scope('layer5'):
            net = slim.flatten(net, scope='flattern')
            net = slim.fully_connected(net, 64, scope='fc6')
            end_point = "layer4"
            end_points[end_point] = net

        with tf.variable_scope('layer6'):
            net = slim.fully_connected(net, 32, scope='fc6')
            end_point = "layer8"
            end_points[end_point] = net

        with tf.variable_scope('layer7'):
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout')
            net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc8')

    return  net, end_points