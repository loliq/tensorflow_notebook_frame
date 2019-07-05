"""
denseNet 实现
denseNet 结构：
    每一个dense_block中的每一个卷积层的输出都是后面所有层的输入
denseNet中包括以下几个部分：
    1. bottleNeck_layer  用于dense_block之前减少输入的feature map数量，降维减少计算量和融合各个通道的特征
                         包括一个1x1的降维模块和一个3x3的卷积模块
    2. transition_layer  用于dense_block之间减小feature map

    3. dense_block  里面的卷积层，每一层的输出都是同一个denseblock里后面每一层的输入
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim

NUM_FILTERS = 12
NUM_BLOCKS = 3


def conv_layer(input, filters, kernel_size, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        net = slim.conv2d(input, filters, kernel_size, stride, scope=layer_name)
        return net


def bottleneck_layer(x, scope):
    # [BN --> ReLU --> conv11 --> BN --> ReLU -->conv33]
    with tf.name_scope(scope):
        # x = slim.batch_norm(x)
        # x = tf.nn.relu(x)
        x = conv_layer(x, NUM_FILTERS, kernel_size=(1, 1), layer_name=scope + '_conv1')
        # x = slim.batch_norm(x)
        # x = tf.nn.relu(x)
        x = conv_layer(x, NUM_FILTERS, kernel_size=(3, 3), layer_name=scope + '_conv2')
        return x


def se_block(x, ratio, scope):
    shape = x.get_shape().as_list()
    channel_out = shape[3]
    # print(shape)
    with tf.variable_scope("squeeze_and_excitation/" + scope):
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


def transition_layer(x, scope):
    # [BN --> conv11 --> avg_pool2]
    with tf.name_scope(scope):
        # x = slim.batch_norm(x)
        x = conv_layer(x, NUM_FILTERS, kernel_size=(1, 1), layer_name=scope + '_conv1')
        x = slim.avg_pool2d(x, [2, 2], stride=2)
        return x


def dense_block(input_x, nb_layers, layer_name):
    with tf.name_scope(layer_name):
        layers_concat = []
        layers_concat.append(input_x)
        x = bottleneck_layer(input_x, layer_name + '_bottleN_' + str(0))
        layers_concat.append(x)
        for i in range(nb_layers):
            x = tf.concat(layers_concat, axis=3)
            x = bottleneck_layer(x, layer_name + '_bottleN_' + str(i + 1))
            layers_concat.append(x)
        # 输出前使用se_block建模通道关系
        x = se_block(x, 4, scope=layer_name)
        return x


def inference(inputs,
              num_classes=2,
              dropout_keep_prob=0.8,
              is_training=True,
              weights_reg=0.005):
    # 刚开始的层，接下来用到slim的都按照这个参数
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(weights_reg),  # 越小惩罚项越重
                        normalizer_fn=tf.layers.batch_normalization,
                        normalizer_params={'training': is_training,
                                           'momentum': 0.95}
                        ):
        #维度缩减
        end_points = {}

        net = conv_layer(inputs, NUM_FILTERS, kernel_size=(7, 7),
                         stride=2, layer_name='layer1_conv0')
        net = se_block(net, 4, scope='conv1')
        end_points['layer1'] = net  #112
        net = slim.max_pool2d(net, (3, 3), stride=2, padding='SAME')  # 56
        num_layers_list = [2, 4, 6]
        # input 56x56
        for i in range(NUM_BLOCKS):
            layer_name = "dense_block" + str(i)
            net = dense_block(net, num_layers_list[i], 'dense_' + str(i))
            net = transition_layer(net, 'trans_' + str(i))
            end_points[layer_name] = net
        net = slim.avg_pool2d(net, kernel_size=[7, 7], stride=1, scope='pool')
        end_points['final_avg_pool'] = net
        net = slim.flatten(net, scope='flattern')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout')
        net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc8')
    return net, end_points




