import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
from create_record_files import create_dataset
from create_record_files import get_example_nums
from slim.nets.mobilenet_v1 import mobilenet_v1_arg_scope, mobilenet_v1
from slim.nets.inception_v4 import inception_v4,inception_v4_arg_scope
import PARAMS as Param
import models.cnn_layers2 as net
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import dataset_factory.dataset_factory as datasets
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import logging
import numpy as np
import PARAMS

file_logger = logging.getLogger('terminal-logger')
file_logger.setLevel(logging.DEBUG)
# termin_logger = logging.getLogger('file-logger')
# termin_logger.setLevel(logging.INFO)
fh = logging.FileHandler('test.md')
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
file_logger.addHandler(fh)
file_logger.addHandler(ch)

params = PARAMS.Params()
class_nums = params.params['model']['classNum']
resize_height = params.params['model']['height'] # 指定存储图片高度
resize_width = params.params['model']['width']  # 指定存储图片宽度
labels_filename='E:/LLL/deepLearning/cell_data//5_fold/label.txt'
batch_size = 1  #
depths = params.params['model']['depth']
tensorBoard_path = params.params['path']['test_tensorBoardPath']

# TODO CHANGE MODEL PATH AND MODEL NAME
image_dirs = 'E:/LLL/deepLearning/cell_data//5_fold/data_set/val/NG'
output_graph = 'shell_model/CNN6_layer2.pb'
models_path = 'shell_model/model_epoch93_0.9515.ckpt'  #查看checkPoint的名称即可
val_dir = os.path.dirname(params.params['path']['val_rex'])
val_num = get_example_nums(val_dir)
print("validation num = {0}".format(val_num))



def test_dataSet(model_name, dataset_dir, process_name, batch_size):
    print_tensors_in_checkpoint_file(models_path,None,False,True)
    test_dataset = datasets.get_dataset(process_name, dataset_dir,
                                        'test', batch_size=batch_size)
    test_iterator = test_dataset.make_initializable_iterator()
    test_images, test_labels = test_iterator.get_next()
    labels = np.loadtxt(labels_filename, str, delimiter='\t')
    with tf.name_scope("inputs"):
        is_training = tf.placeholder(tf.bool)
        images = tf.placeholder(
            dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='inputs')
        labels = tf.placeholder(dtype=tf.int32, shape=[None, class_nums], name='label')

    with tf.name_scope("net"):
        logits,endpoints = net.inference(inputs=images, num_classes=class_nums,
                               is_training=is_training, dropout_keep_prob=1.0)

    # 将输出结果进行softmax分布,再求最大概率所属类别
    with tf.name_scope('Output'):
        score = tf.nn.softmax(logits,name='predict')
        class_id = tf.argmax(score, 1)
        correct_prediction = tf.equal(class_id, tf.argmax(labels, 1))
    saver = tf.train.Saver()
    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    layer5_mv = tf.get_default_graph().get_tensor_by_name('layer4/conv/conv_2/batch_normalization/moving_mean:0')
    layer5_va = tf.get_default_graph().get_tensor_by_name('layer4/conv/conv_2/batch_normalization/moving_variance:0')

    with tf.Session() as sess:

        sess.run(init_op)
        saver.restore(sess, model_name)
        sess.run(test_iterator.initializer)
        num_correct, num_samples = 0, 0
        while True:
            try:
                test_batch_images, test_batch_labels \
                    = sess.run([test_images, test_labels])
                score_val, correct_pred,class_index \
                    = sess.run([ score,correct_prediction,class_id],
                               feed_dict={is_training: False,
                                          images: test_batch_images,
                                           labels: test_batch_labels})
                mv_val, va_val = sess.run([ layer5_mv,layer5_va],
                                       feed_dict={is_training: False,
                                                  images: test_batch_images,
                                                   labels: test_batch_labels})

                num_correct += correct_pred.sum()
                num_samples += correct_pred.shape[0]
                # print("mv{0}".format(mv_val))
                # print("va{0}".format(va_val))
                # print(str(score_val)+" " + str(class_index))
            except tf.errors.OutOfRangeError:
                break

        acc = float(num_correct) / num_samples
        return acc
if __name__ == '__main__':
    acc = test_dataSet(models_path, val_dir, 'iwatch',1)
    print("test acc is %f" % acc)