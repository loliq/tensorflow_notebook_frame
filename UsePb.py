"""
     -*- coding: utf-8 -*-
    @Project: PyCharm
    @File    : Use_pb.py
    @Author  : LLL
    @Site    :
    @Email   : lilanluo@stu.xmu.edu.cn
    @Date    : 2019/3/12 16:11
    @info   :
    -  给定pb模型路径及名称, 图片文件夹路径对测试集进行单张图像的前向测试
    -
"""
import os, argparse
import numpy as np
import tensorflow as tf
from tfRecord_func import *

def print_all_node_name():
    """
    打印默认图的所有节点名称
    :return:
    """
    for n in tf.get_default_graph().as_graph_def().node:
            print(n.name)

def freeze_graph(model_dir, output_node_names):
    """
    输入网络文件，冻结成pb,实际使用的时候在测试的时候存成pb比较好
    :param model_dir:
    :param output_node_names:
    :return:
    """
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath  //检索checkPoint的完整路径
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph 绝对模型路径
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1]) # [:-1]的意思是获取最后一个切片
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def

def load_graph(frozen_graph_filename):
    """
    输入pb，返回重命名后的图
    :param frozen_graph_filename:
    :return:
    """
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="")  #加上了name不行，具体看
    return graph

def Predict(image_path, PbFileName, resize_height,resize_width):
    """
    测试使用pbFile，单次读取一张图片并输出结果，
    注意要给定输入节点的名称，用sess.graph.get_tensor_by_name("input:0")，节点格式为"nodeName:index"
    :param image_path: 图像路径
    :param PbFileName: pb图像路径
    :param resize_height: 图像高度
    :param resize_width: 图像宽度
    :return:
    """
    #读取pbFile
    labels_filename = 'dataset/label.txt'
    output_graph_def = tf.GraphDef()
    labels = np.loadtxt(labels_filename, str, delimiter='\t')
    with open(PbFileName, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name="")
    init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        #TODO REDEFINE THE INPUT TENSOR IF NESSASARY
        # 定义输入的张量名称,对应网络结构的输入张量
        # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
        input_image_tensor = sess.graph.get_tensor_by_name("Input/input_image:0")
        output_tensor_name = sess.graph.get_tensor_by_name("Output/predict:0")
        input_imge = read_image(image_path, resize_height, resize_width, normalization=True)
        input_imge = sess.run(input_imge)
        input_imge = input_imge[np.newaxis, :]

        score = sess.run(output_tensor_name, feed_dict={input_image_tensor: input_imge
                                                      })

        print("score:{}".format(score))
        class_id = tf.argmax(score, 1)
        print( "pre class is :{}".format(labels[sess.run(class_id)]))


if __name__ == '__main__':
    #TODO change the path
    image_path = 'dataSet/test/NG/287ng.bmp'
    PbFileName = 'iwatch_model/frozen_model.pb'
    Predict(image_path, PbFileName, 64,64)