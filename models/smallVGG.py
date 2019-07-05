import tensorflow as tf
import tflearn as tfl
import tensorflow.contrib.slim as slim
CONVSIZE1 = 3

def inference(input_tensor, train, regularizer=None):

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn = tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        normalizer_fn = slim.batch_norm):
        #等于下面注释的代码

        conv12 = slim.repeat(input_tensor, 2, slim.conv2d, 64, [CONVSIZE1, CONVSIZE1], scope='conv12')
        pool1 = slim.max_pool2d(conv12, kernel_size=[2, 2], stride=2, scope='pool1')
        #第二层定义
        conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [CONVSIZE1, CONVSIZE1], scope='conv2')
        pool2 = slim.max_pool2d(conv2, kernel_size=[2,2], stride=2, scope='pool2')

        # #第三层
        layer2Decrease = slim.conv2d(pool2, 64, [1, 1], scope='Rconv23')
        conv3 = slim.repeat(layer2Decrease,2,slim.conv2d, 256, [CONVSIZE1, CONVSIZE1], scope='conv3')
        pool3 = slim.max_pool2d(conv3, kernel_size=[2, 2], stride=2, scope='pool3')


        pool_shape = pool3.get_shape().as_list()

        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool3, [pool_shape[0], nodes])

        fc6 = slim.fully_connected(reshaped, 512, scope='fc6')
        if train:
            dropout7 = slim.dropout(fc6, 0.3, scope='dropout7')
        else:
            dropout7 = fc6
        fc8_Out = slim.fully_connected(dropout7, 2, activation_fn=None, scope='fc8')
    return  fc8_Out