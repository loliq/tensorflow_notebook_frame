"""
     -*- coding: utf-8 -*-
    @Project: PyCharm
    @File    : test_Predict.py
    @Author  : LLL
    @Site    :
    @Email   : lilanluo@stu.xmu.edu.cn
    @Date    : 2019/3/12 16:11
    @info   :
    -  给定模型路径及名称, 图片文件夹路径对测试集进行单张图像测试
    -  给定pb模型文件存成pb文件
    -
"""

import tensorflow as tf
import os
import glob
import models.cnn_layers2 as net
import numpy as np
import logging
import PARAMS
from dataset_factory.iwatch import process_image_convert
from create_record_files import read_image
import tensorflow.contrib.slim as slim


file_logger = logging.getLogger('terminal-logger')
file_logger.setLevel(logging.DEBUG)
# termin_logger = logging.getLogger('file-logger')
# termin_logger.setLevel(logging.INFO)
fh = logging.FileHandler('test.md')
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
file_logger.addHandler(fh)
file_logger.addHandler(ch)
# termin_logger.addHandler(ch)


params = PARAMS.Params()
class_nums = params.params['model']['classNum']
resize_height = params.params['model']['height'] # 指定存储图片高度
resize_width = params.params['model']['width']  # 指定存储图片宽度
labels_filename='E:/LLL/deepLearning/cloth_data/5_fold/label.txt'
batch_size = 1  #
depths = params.params['model']['depth']
tensorBoard_path = params.params['path']['test_tensorBoardPath']

# TODO CHANGE MODEL PATH AND MODEL NAME
image_dirs = 'E:/LLL/deepLearning/cloth_data/5_fold/5_of_5/NG'
output_graph = 'shell_model/CNN6_layer2.pb'
models_path = 'shell_model/model_epoch27_0.9619.ckpt'  #查看checkPoint的名称即可

def  predict(models_path,image_dir,labels_filename,labels_nums, data_format,is_savePb=False):
    [batch_size, resize_height, resize_width, depths] = data_format

    labels = np.loadtxt(labels_filename, str, delimiter='\t')
    with tf.name_scope("inputs"):
        input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths],
                                      name='inputs')  # 由于图像存储的原因，灰度图维度较少一维
        is_training = tf.placeholder(tf.bool, name='is_training')
    with tf.name_scope("net"):
        out,endpoints = net.inference(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=1.0,
                            is_training=is_training)

    # 将输出结果进行softmax分布,再求最大概率所属类别
    with tf.name_scope('Output'):
        score = tf.nn.softmax(out,name='predict')
        class_id = tf.argmax(score, 1)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, models_path)
    images_list=glob.glob(os.path.join(image_dir,'*.bmp'))
    for i in range(len(images_list)):
        image_string = tf.read_file(images_list[i])
        image_decoded = tf.image.decode_bmp(image_string, channels=1)  # (1)
        image = tf.cast(image_decoded, tf.float32)
        input_image = process_image_convert(image,is_training=False)
        input_image = tf.expand_dims(input_image,0)
        img_real = sess.run(input_image)
        pre_score,pre_label = sess.run([score,class_id], feed_dict={input_images:img_real,
                                                                    is_training:False
                                                                    })
        max_score=pre_score[0,pre_label]
        file_logger.info("![]({0}){1} pre labels:{2} score: {3}{4}".format(images_list[i],"  \r",labels[pre_label], max_score,"  \r"))
    if is_savePb:
        ##存成pbfile, 注意必须为List才能存
        outPut_nodeName = []
        outPut_nodeName.append('Output/predict')
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names=outPut_nodeName  # The output node names are used to select the usefull nodes
        )

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
    sess.close()


if __name__ == '__main__':
    data_format=[batch_size,resize_height,resize_width,depths]
    predict(models_path,image_dirs, labels_filename, class_nums, data_format,  is_savePb=True)
    #TODO 记录要做的事情