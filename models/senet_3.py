import tensorflow as tf
import tensorflow.contrib.slim as slim

def SE_block(x,ratio):
    shape = x.get_shape().as_list()
    channel_out = shape[3]
    # print(shape)
    with tf.variable_scope("squeeze_and_excitation"):
        # 全局平均池化层--squeeze操作
        # kernel_size=[H,W],strides=[1, H, W, 1]
        # ouput- batchSize X 1 X 1 X C
        squeeze = tf.nn.avg_pool(x, [1, shape[1], shape[2], 1],
                                 [1, shape[1], shape[2], 1], padding="SAME")
        flattern = slim.flatten(squeeze)
        # 全连接层-  Exacitation操作,使用1X1卷积替代全连接层的作用
        #降维 ouput- batchSize X C/r
        # excitation1_output = slim.conv2d(squeeze,channel_out/ratio, kernel_size=[1,1],
        #                                  activation_fn=tf.nn.relu, normalizer_fn=None)
        excitation1_output = slim.fully_connected(flattern, int(channel_out/ratio),
                                                  activation_fn=tf.nn.relu, normalizer_fn=None)
        #升维 ouput- batchSize X C
        # excitation2_output = slim.conv2d(excitation1_output, channel_out, kernel_size=[1, 1],
        #                                  activation_fn=tf.nn.sigmoid, normalizer_fn=None)
        excitation2_output = slim.fully_connected(excitation1_output, channel_out,
                                                  activation_fn=tf.nn.sigmoid, normalizer_fn=None)
        # 第四层，点乘 reshape成
        excitation_output = tf.reshape(excitation2_output, [-1, 1, 1, channel_out])
        h_output = excitation_output * x  #python广播
    return h_output

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
                        normalizer_fn=tf.layers.batch_normalization,
                        normalizer_params={'training': is_training,
                                           'momentum': 0.95}
                        ):
        # 等于下面注释的代码
        # 等于下面注释的代码
        # 224x224x3 ->56x56x16
        end_points = {}
        with tf.variable_scope('layer1'):
            net = slim.repeat(inputs, 2, slim.conv2d, 16, [3, 3], scope='conv')
            net = SE_block(net, 4)
            net = slim.max_pool2d(net, kernel_size=[4, 4], stride=4, scope='pool')  # 32
            end_point = "layer1"
            end_points[end_point] = net
        # 第二层定义
        # 56x56x32 ->14x14x64
        with tf.variable_scope('layer2'):
            net = slim.repeat(net, 2, slim.conv2d, 32, [3, 3], scope='conv')
            net = slim.max_pool2d(net, kernel_size=[4, 4], stride=4, scope='pool')  # 16
            net = SE_block(net, 4)
            end_point = "layer2"
            end_points[end_point] = net

        # #第三层
        # 14x14x128->3x3
        with tf.variable_scope('layer3'):
            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope='conv')
            net = SE_block(net, 4)
            net = slim.max_pool2d(net, kernel_size=[4, 4], stride=4, scope='pool')  # 8
            end_point = "layer3"
            end_points[end_point] = net

        with tf.variable_scope('layer4'):
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv')
            net = SE_block(net, 4)
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=1, scope='pool')
            end_point = "layer3"
            end_points[end_point] = net

        with tf.variable_scope('layer5'):
            net = slim.flatten(net, scope='flattern')
            # net = slim.fully_connected(net, 64, scope='fc6')
            # end_point = "layer4"
            # end_points[end_point] = net

        # with tf.variable_scope('layer6'):
        #     net = slim.fully_connected(net, 64, scope='fc6')
        #     end_point = "layer8"
        #     end_points[end_point] = net

        with tf.variable_scope('layer7'):
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout')
            net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc8')

    return net, end_points


