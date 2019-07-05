import tensorflow as tf
import tflearn as tfl
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils


def inference(inputs,
          num_classes = 2,
          dropout_keep_prob=0.8,
          is_training=True,
          scope = 'vgg_a',
          fc_conv_padding='VALID',
          global_pool=False):
  """Oxford Net VGG 11-Layers version A Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_a', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        normalizer_fn=slim.batch_norm,
                        outputs_collections=end_points_collection):
      end_points = {}
      # 224->112
      with tf.variable_scope('layer1'):
        net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv')
        net = slim.max_pool2d(net, [2, 2], scope='pool')
        end_point = "layer1"
        end_points[end_point] = net
      # 112 ->56
      with tf.variable_scope('layer2'):
        net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv')
        net = slim.max_pool2d(net, [2, 2], scope='pool')
        end_point = "layer2"
        end_points[end_point] = net
      # 56-28
      with tf.variable_scope('layer3'):
        net = slim.conv2d(net, 64, [1, 1], scope='Rconv')
        net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv')
        net = slim.max_pool2d(net, [2, 2], scope='pool')
        end_point = "layer3"
        end_points[end_point] = net
      # 28->14
      with tf.variable_scope('layer4'):
        net = slim.conv2d(net, 128, [1, 1], scope='Rconv')
        net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv')
        net = slim.max_pool2d(net, [2, 2], scope='pool')
        end_point = "layer4"
        end_points[end_point] = net
      # 14 ->7
      with tf.variable_scope('layer5'):
        net = slim.conv2d(net, 128, [1, 1], scope='Rconv')
        net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        end_point = "layer5"
        end_points[end_point] = net
      # Use conv2d instead of fully_connected layers.
      with tf.variable_scope('layer6'):
        net = slim.conv2d(net, 512, [7, 7], padding=fc_conv_padding, scope='con6')
        end_point = "layer6"
        end_points[end_point] = net
      with tf.variable_scope('layer7'):
        net = slim.flatten(net)
        net = slim.fully_connected(net,256, scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
      with tf.variable_scope('layer8'):
        net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc8')
      return net,end_points

