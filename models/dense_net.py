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
        return x

def inference(inputs,
              num_classes=2,
              dropout_keep_prob=0.8,
              is_training=True):
    # 刚开始的层，接下来用到slim的都按照这个参数
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(0.00001),  # 越小惩罚项越重
                        normalizer_fn=tf.layers.batch_normalization,
                        normalizer_params={'training': is_training,
                                           'momentum': 0.95}
                        ):
        #维度缩减
        end_points = {}

        net = conv_layer(inputs, NUM_FILTERS, kernel_size=(7, 7),
                         stride=2, layer_name='layer1_conv0')
        end_points['layer1'] = net
        net = slim.max_pool2d(net, (3, 3), stride=2, padding='SAME')
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




