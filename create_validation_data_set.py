from segment_datasets import gen_data_txt,write_label_text,trans_dir,gen_data_txt_with_labels
import os
import numpy as np
import shutil

def segment_k_fold(data_path, dst_path, num_fold):
    """

    :param data_path:
    :param num_fold:
    :return:
    """
    data_list= os.path.join(data_path, 'data.txt')
    with open(data_list, 'r') as f:
        lines = f.readlines()
        file_index_array = np.arange(0,len(lines))
        np.random.shuffle(file_index_array)  # 生成随机序列
        labels = write_label_text(data_path, dst_path)
        total_sample_num = len(file_index_array)
        per_fold = 1.0 / num_fold
        for fold_index in range(num_fold):
            fold_dir_name = "{0}_of_{1}".format(fold_index+1, num_fold)
            fold_path_name = os.path.join(dst_path, fold_dir_name)
            if not os.path.exists(fold_path_name):
                os.makedirs(fold_path_name)
                for label in labels:
                    os.makedirs(os.path.join(fold_path_name, label))

            data_start = 0
            data_end = 0
            if fold_index == (num_fold-1):
                data_start = int(np.floor((fold_index)*per_fold*total_sample_num))
                data_end = total_sample_num - 1
            else:
                data_start = int(np.floor(fold_index*per_fold*total_sample_num))
                data_end = int(np.floor((fold_index + 1)*per_fold*total_sample_num)-1)
            print("data start" + str(data_start))
            print("data end" + str(data_end))
            lines_array= file_index_array[data_start:data_end]
            print(lines_array)
            for line_index in lines_array:
                line = lines[line_index].strip('\n')
                shutil.copy(os.path.join(data_path, line), os.path.join(fold_path_name, line))

def compose_to_dataset(folder_pathes, dst_path):

    for folder in folder_pathes:
       subdirs =  os.listdir(folder)
       for subdir in subdirs:
           src_folder = os.path.join(folder, subdir)
           if os.path.isdir(src_folder):
               dst_folder = os.path.join(dst_path, os.path.basename(subdir))
               if not os.path.exists(dst_folder):
                   os.makedirs(dst_folder)
               for file in os.listdir(os.path.join(folder, subdir)):
                  shutil.copy(os.path.join(src_folder, file),
                              os.path.join(dst_folder, file))
               # shutil.copytree(folder, os.path.join(dst_folder))


if __name__ == '__main__':


    data_path = trans_dir("E:\LLL\deepLearning/03_iwatch_line_scan/0609iwatch2")
    # gen_data_txt(data_path)
    # #
    dst_path = trans_dir("E:\LLL\deepLearning/03_iwatch_line_scan/5_fold")
    dst_train = dst_path + '/data_set/train'
    dst_val = dst_path + 'data_set/val'
    # label = write_label_text(data_path, dst_path)
    # segment_k_fold(data_path, dst_path, 5 )
    # #
    val_name = "1_of_5"
    nameList = ["{0}_of_5".format(i+1) for i in range(5)]
    nameList.remove(val_name)
    nameDir = [os.path.join(dst_path, name) for name in nameList]

    print(nameDir)
    print(dst_train)
    compose_to_dataset(nameDir, dst_train)
    gen_data_txt_with_labels(dst_train,
                             "train.txt",
                             os.path.join(dst_path, 'label.txt'))

    shutil.copytree(os.path.join(dst_path, val_name),
                    dst_val)
    gen_data_txt_with_labels(dst_val,
                             "val.txt",
                             os.path.join(dst_path, 'label.txt'))
