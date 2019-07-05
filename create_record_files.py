"""
     -*- coding: utf-8 -*-
    @Project: PyCharm
    @File    : create_record_files.py
    @Author  : LLL
    @Site    : 
    @Email   : lilanluo@stu.xmu.edu.cn
    @Date    : 2019/5/13 9:41
    @info   :
"""
import tensorflow as tf
import numpy as np
import PARAMS as Params
import os
import matplotlib.pyplot as plt
import random
from PIL import Image
import io
from segment_datasets import trans_dir



def create_dataset(filenames, batch_size=8, is_shuffle=False, n_repeats=0):
    """

    :param filenames: record file names
    :param batch_size:
    :param is_shuffle: 是否打乱数据
    :param n_repeats:
    :return:
    """
    dataset = tf.data.TFRecordDataset(filenames)
    if n_repeats > 0:
        dataset = dataset.repeat(n_repeats)         # for train
    if n_repeats == -1:
        dataset = dataset.repeat()  # for val to
    dataset = dataset.map(lambda x: parse_single_exmp(x, labels_nums=2))
    if is_shuffle:
        dataset = dataset.shuffle(10000)            # shuffle
    dataset = dataset.batch(batch_size)
    return dataset

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value])) # if Value is not list,then add[]


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 生成实数型的属性
def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_example_nums(tf_records_dir):
    """

    统计tf_records图像的个数(example)个数
    :param tf_records_dir: tf_records文件路径
    :return:
    """
    files = os.listdir(tf_records_dir)
    print(files)
    nums = 0
    for file in files:
        for record in tf.python_io.tf_record_iterator(os.path.join(tf_records_dir, file)):
            nums += 1
    return nums


def load_labels_file(filename, labels_num=1, shuffle=False):
    """
    载图txt文件，文件中每行为一个图片信息，且以空格隔开：图像路径 标签1 标签2，如：test_image/1.jpg 0 2
    :param filename:
    :param labels_num :labels个数
    :param shuffle :是否打乱顺序
    :return:images type->list
    :return:labels type->list
    """
    images = []
    labels = []
    with open(filename) as f:
        lines_list = f.readlines()
        if shuffle:
            random.shuffle(lines_list)
        for lines in lines_list:
            line = lines.rstrip().split(' ')  #空格分割
            label = []
            for i in range(labels_num):
                label.append(int(line[i+1]))
            images.append(line[0])
            labels.append(label)
    return images, labels


def show_image(title,image):
    """
     显示图片
    :param title: 图像标题
    :param image: 图像的数据
    :return:
    """
    # plt.figure("show_image")
    # print(image.dtype)
    plt.imshow(image)
    plt.axis('on')    # 关掉坐标轴为 off
    plt.title(title)  # 图像题目
    plt.show()


def read_image(filename, resize_height, resize_width,channel = 1):
    """
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :return: 返回的图片数据
    """
    img= Image.open(filename,'r')
    size = img.size
    # img_raw = img.tobytes()
    return  img,size


def create_records(image_dir, file, output_record_dir, file_name, instances_per_shard,
                   resize_height, resize_width, shuffle=True, log=5):
    """
    实现将图像原始数据,label,长,宽等信息保存为record文件
    注意:读取的图像数据默认是uint8,再转为tf的字符串型BytesList保存,解析请需要根据需要转换类型
    :param image_dir:原始图像的目录
    :param file:输入保存图片信息的txt文件(image_dir+file构成图片的路径)
    :param output_record_dir:保存record文件的路径
    :param file_name:保存record文件的路径
    :param instances_per_shard:保存record文件的路径
    :param resize_height:
    :param resize_width:
    PS:当resize_height或者resize_width=0是,不执行resize
    :param shuffle:是否打乱顺序
    :param log:log信息打印间隔
    """
    # 加载文件,仅获取一个label
    images_list, labels_list=load_labels_file(file,1,shuffle)
    num_example = len(images_list)
    num_train_shards = int(num_example / instances_per_shard)

    for index, [image_name, labels] in enumerate(zip(images_list, labels_list)):    #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        if index == 0:
            filename = (output_record_dir + '/' + file_name + '-%.2d-of-%.2d' % (0, num_train_shards))
            print(filename)
            writer = tf.python_io.TFRecordWriter(filename)
        if index % instances_per_shard == 0 and index != 0:
            writer.close()
            tf_index = index / instances_per_shard
            filename = (output_record_dir + '/'+file_name+'-%.2d-of-%.2d' % (tf_index, num_train_shards))
            print(filename)
            writer = tf.python_io.TFRecordWriter(filename)
        image_path=os.path.join(image_dir,image_name)
        print(image_path)
        if not os.path.exists(image_path):
            print('Err:no image',image_path)
            continue
        image,size = read_image(image_path, resize_height, resize_width)
        image = image.tobytes()
        # image_bytes = image.toString()
        if index % log == 0 or index == len(images_list)-1:
            print('------------processing:%d-th------------' % (index))
            print('current image_path=%s' % (image_path),'shape:{}'.format(size),'labels:{}'.format(labels))
        # 这里仅保存一个label,多label适当增加"'label': _int64_feature(label)"项
        label=labels[0]
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def read_records(tf_data_regrex,resize_height, resize_width, shuffle=True, type = None, opposite=True, channel=1,numepoch=None):
    """
    解析record文件:源文件的图像数据是,uint8,[0,255],一般作为训练数据时,需要归一化到[0,1]
    :param tf_data_regrex: 匹配tfRecord File的正则化表达式
    :param resize_height: 图像高度
    :param resize_width: 图像宽度
    :param shuffle: 是否打乱数据
    :param type:选择图像数据的返回类型
         None:默认将uint8-[0,255]转为float32-[0,255]
         normalization:归一化float32-[0,1]
         centralization:归一化float32-[0,1],再减均值中心化
    :param opposite:
    :param channel:
    :param numepoch:
    :return:
    """
    # 创建文件队列,不限读取的数量
    files = tf.train.match_filenames_once(tf_data_regrex)
    filename_queue = tf.train.string_input_producer(files, num_epochs=numepoch, shuffle=shuffle)
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )

    tf_image = tf.decode_raw(features['image_raw'], tf.uint8)#获得图像原始的数据
    tf_height = features['height']
    tf_width = features['width']
    tf_depth = features['depth']
    tf_label = tf.cast(features['label'], tf.int32)
    # PS:恢复原始图像数据,reshape的大小必须与保存之前的图像shape一致,否则出错
    # tf_image=tf.reshape(tf_image, [-1])    # 转换为行向量
    tf_image=tf.reshape(tf_image, [resize_height, resize_width, channel]) # 设置图像的维度
    tf_image = tf.cast(tf_image, tf.float32)
    # 制作tf_record时已经归一化了
    tf_image = prepeocess(
        tf_image, choice=opposite)
    return tf_image, tf_label


def get_batch_images(images,labels,batch_size,labels_nums,one_hot=True,shuffle=False,num_threads=1):
    """
    :param images:图像
    :param labels:标签
    :param batch_size:
    :param labels_nums:标签个数
    :param one_hot:是否将labels转为one_hot的形式
    :param shuffle:是否打乱顺序,一般train时shuffle=True,验证时shuffle=False
    :return:返回batch的images和labels
    """
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size  # 保证capacity必须大于min_after_dequeue参数值
    if shuffle:
        # 调用参数过长时首行不显示参数
        images_batch, labels_batch = tf.train.shuffle_batch(
            [images, labels], batch_size=batch_size,
            capacity=capacity, min_after_dequeue=min_after_dequeue, num_threads=num_threads)
    else:
        images_batch, labels_batch = tf.train.batch(
            [images, labels],batch_size=batch_size, capacity=capacity, num_threads=num_threads)
    if one_hot:
        labels_batch = tf.one_hot(labels_batch, labels_nums, 1, 0)  # 可以直接转成one-hot-label
    images_batch = tf.expand_dims(images_batch, -1)
    return images_batch,labels_batch


def disp_records(record_file, resize_height, resize_width, show_nums=4):
    """
    解析record文件，并显示show_nums张图片，主要用于验证生成record文件是否成功
    :param record_file: record文件路径
    :param resize_height,
    :param resize_width,
    :param resize_width,
    :param show_nums
    :return:
    """
    # 读取record函数
    tf_image, tf_label = read_records(record_file, resize_height, resize_width,type='normalization')
    # 显示前4个图片
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) #必须一起初始化否则会报错
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(show_nums):
            image, label = sess.run([tf_image, tf_label])  # 在会话中取出image和label
            # image = tf_image.eval()
            # 直接从record解析的image是一个向量,需要reshape显示
            # image = image.reshape([height,width,depth])
            print('shape:{},tpye:{},labels:{}'.format(image.shape, image.dtype, label))
            show_image("image:%d" % (label), image)
        coord.request_stop()
        coord.join(threads)


def parse_single_exmp(serialized_example,labels_nums=2):
    """
    解析tf.record
    :param serialized_example:
    :param opposite: 是否将图片取反
    :return:
    """
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    tf_image = tf.decode_raw(features['image_raw'], tf.uint8)#获得图像原始的数据
    tf_height = features['height']
    tf_width = features['width']
    tf_depth = features['depth']
    tf_label = tf.cast(features['label'], tf.int32)
    # PS:恢复原始图像数据,reshape的大小必须与保存之前的图像shape一致,否则出错
    # tf_image=tf.reshape(tf_image, [-1])    # 转换为行向量
    tf_image =tf.reshape(tf_image, [224, 224, 3]) # 设置图像的维度
    tf_image = tf.cast(tf_image, tf.float32)
    tf_image = preprocess(
        tf_image, choice=True)
    tf_label = tf.one_hot(tf_label, labels_nums, 1, 0)
    print(tf_image)
    return tf_image, tf_label


def batch_test(record_file,labels_nums, resize_height, resize_width):
    """
    :param record_file: record文件路径
    :param labels_nums
    :param resize_height:
    :param resize_width:
    :return:
    :PS:image_batch, label_batch一般作为网络的输入
    """
    # 读取record函数
    tf_image,tf_label = read_records(record_file,resize_height,resize_width,type='normalization')
    image_batch, label_batch = get_batch_images(tf_image,tf_label,batch_size=64,labels_nums=labels_nums,one_hot=True,shuffle=True)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:  # 开始一个会话
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(4):
            # 在会话中取出images和labels
            images, labels = sess.run([image_batch, label_batch])
            print('shape:{},tpye:{},labels:{}'.format(images.shape,images.dtype,labels))

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    # 参数设置
    for validation_index in range(5):
        root_dir = 'dataset/dataset/{0}_of_5folds'.format(validation_index+1)
        root_dir = trans_dir(root_dir)
        PARAMS = Params.Params()
        resize_height = PARAMS.params['model']['height']  # 指定存储图片高度
        resize_width = PARAMS.params['model']['width']  # 指定存储图片宽度
        train_regrex = "dataset/record_file/iwatch_224_record_v{0}/train/train-*".format(validation_index+1)
        val_regrex = "dataset/record_file/iwatch_224_record_v{0}/val/val-*".format(validation_index+1)
        dataSet_regrex = [train_regrex, val_regrex]
        for regrexName in dataSet_regrex:
            path = os.path.dirname(regrexName)
            if not os.path.exists(path):
                os.makedirs(path)
        shuffle = True
        log = 5
        instances_per_shard = 500
        # 产生train.record文件
        image_dir = root_dir + '/train'
        train_labels = root_dir + '/train.txt'  # 图片路径
        train_record_output = os.path.dirname(train_regrex)
        create_records(image_dir, train_labels, train_record_output, 'train', instances_per_shard, resize_height,
                       resize_width,shuffle, log=5)
        # # # 产生test.record文件
        image_dir = root_dir + '/val'
        val_labels = root_dir + '/val.txt'  # 图片路径
        val_record_output = os.path.dirname(val_regrex)
        create_records(image_dir, val_labels, val_record_output, 'val', instances_per_shard, resize_height, resize_width,
                       shuffle, log=5)
        val_nums = get_example_nums(val_record_output)
        print("save val example nums={}".format(val_nums))
        train_nums=get_example_nums(train_record_output)
        print("save train example nums={}".format(train_nums))
