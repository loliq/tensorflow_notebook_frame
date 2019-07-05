"""
     -*- coding: utf-8 -*-
    @Project: PyCharm
    @File    : Segment_DataSets.py
    @Author  : LLL
    @Site    : 
    @Email   : lilanluo@stu.xmu.edu.cn
    @Date    : 2019/5/13 9:51
    @info   :分割图片为训练集,验证集和测试集
"""

import os
import shutil
import re
import numpy as np


# 原数据集
SRCPATH = "E:\LLL\deepLearning\halcon18\iwatch\data031101"
# 整理后的数据集
DSTPATH = 'dataset'
TRAIN_PERCENT = 0.8
VAL_PERCENT = 0.10


def remove_dirs(folder_path):
    """
     删除文件夹
    :param folder_path:
    :return:
    """
    current_filelist = os.listdir(folder_path)
    for floor1 in current_filelist:
       if os.path.isdir(os.path.join(folder_path,floor1)):
          real_folder_path = os.path.join(folder_path, floor1)
          for root, dirs, files in os.walk(real_folder_path):
             for name in files:
                del_file = os.path.join(root, name)
                os.remove(del_file)


def trans_dir(oriDir):
    transformed_dir = re.sub(r'\\', '/', oriDir)
    return transformed_dir


def gen_data_txt(rootdir):
    """
    生成元数据集的索引
    :param rootdir:
    :return:
    """
    sub_dirs = [x[0] for x in os.walk(rootdir)]
    del sub_dirs[0]
    with open(os.path.join(rootdir,"data.txt"), 'w') as f:
        base_names = os.path.basename(sub_dirs[0])
        print(base_names)
        for sub_dir in sub_dirs:
            base_name = os.path.basename(sub_dir)
            list0 = os.listdir(sub_dir)
            for j in range(0, len(list0)):
                f.write(base_name + "/" + list0[j] + '\n')

def create_dir(src_path, data_path, delete = True):

    """
    判断文件夹是否存在，删除文件夹的内容
    :param dataPath:  数据集根目录
    :param labels:
    :param delete:
    :return:   ###注意os.mkdir会报错要用os.makedirs
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if delete is True:
        shutil.rmtree(data_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    labels = write_label_text(src_path, data_path)     # 写label文件
    dataset_labels = ['train','val', 'test']
    for dataset_label in dataset_labels:
        dir_path1 = os.path.join(data_path, dataset_label)
        for label in labels:
            dir_path = os.path.join(dir_path1, label)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)


#写label名称
def write_label_text(src_path, dst_path):
    """
    写标签数据
    :param src_path: 文件夹根目录，目录下为分类的文件夹
    :return:
    """
    labels = []
    for file in os.listdir(src_path):
        if os.path.isdir(os.path.join(src_path, file)):
            labels.append(file)
    if len(labels):
        with open(os.path.join(dst_path, 'label.txt'), 'w') as f:
            for label in labels:
                f.write(label + '\n')
    return labels


def separate_data(src_path, dst_path, trainPer=0.7, valPer =0.15):
    """
    复制数据并分成验证集,测试集和训练集.并写成txt文件
    :param src_path:
    :param dst_path:
    :param trainPer:
    :param valPer:
    :return:
    """
    data_path = os.path.join(src_path, 'data.txt')
    if os.path.isfile(data_path):
        with open(data_path,'r') as f:
            lines = f.readlines()
            file_index_array = np.random.permutation(len(lines)) #生成随机序列
            total_num = len(lines)
            train_num = int(np.floor(total_num * trainPer))
            val_num = int(np.floor(total_num * valPer))
            test_num = total_num - train_num - val_num
            train_index = file_index_array[0:train_num-1]
            val_index = file_index_array[train_num: total_num-test_num-1]
            test_index = file_index_array[total_num-test_num:total_num-1]
            dataset_indexes = [train_index, val_index, test_index]
            dataset_indexes = list(dataset_indexes)
            dataset_names = ['train', 'val', 'test']
            dataset_names = list(dataset_names)
            labels = write_label_text(src_path, dst_path)

            for i in range(3):
                print(dataset_names[i])
                dataset_path = os.path.join(dst_path, dataset_names[i])
                with open(os.path.join(dst_path,dataset_names[i] + '.txt'), 'w') as fileWriter:
                    for index in dataset_indexes[i]:
                        line = lines[index].strip('\n')
                        shutil.copy(os.path.join(src_path, line), os.path.join(dataset_path, line))
                        label_name = os.path.dirname(line)
                        #找出索引
                        label_index = labels.index(label_name)
                        fileWriter.write(line + " " + str(label_index) + '\n')

def gen_data_txt_with_labels(rootdir, fileName, label_file):
    """
    生成元数据集的索引
    :param rootdir:
    :return:
    """
    parent_dir = os.path.dirname(rootdir)
    label_list = []
    if os.path.exists(label_file):
        with open(label_file,'r') as f:
           label_list = f.readlines()
           label_list = [ label.strip('\n') for label in label_list ]


    class_folders = os.listdir(rootdir)
    class_paths = []
    for class_folder in class_folders:
        folder_path = os.path.join(rootdir,class_folder)
        if os.path.isdir(folder_path):
            class_paths.append(folder_path)

    with open(os.path.join(parent_dir, fileName),'w') as f:
        for class_folder in class_folders:
            folder_path = os.path.join(rootdir, class_folder)
            if os.path.isdir(folder_path):
                image_files = os.listdir(folder_path)
                for image_file in image_files:
                    f.write("{0}/{1} {2}\n".format(class_folder, image_file,
                                                 label_list.index(class_folder)))



if __name__ == '__main__':
    SRCPATH = trans_dir(SRCPATH)
    gen_data_txt(SRCPATH)  # 生成data.txt
    create_dir(SRCPATH, DSTPATH, False)
    # write_label_text(SRCPATH, DSTPATH)
    separate_data(SRCPATH, DSTPATH, trainPer=TRAIN_PERCENT, valPer=VAL_PERCENT)
