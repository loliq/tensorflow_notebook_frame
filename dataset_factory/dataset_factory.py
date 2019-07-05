from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dataset_factory import clothes
from dataset_factory import shell
from dataset_factory import iwatch
import os
import tensorflow as tf



datasets_map = {
    'iwatch': iwatch,
    'shell': shell,
    'clothes': clothes,
}


def create_dataset(filenames,
                   batch_size=8,
                   is_shuffle=False,
                   is_training=True,
                   shuffle_num=10000,
                   resize_height=224,
                   resize_width=224,
                    n_repeats=0,
                    label_num = 2,
                   dataset_name='iwatch'):
    """
    :param filenames: record file names
    :param batch_size:
    :param is_shuffle: 是否打乱数据
    :param n_repeats:
    :return:
    """
    dataset = tf.data.TFRecordDataset(filenames)
    if n_repeats > 0:
        dataset = dataset.repeat(n_repeats)
    if n_repeats == -1:
        dataset = dataset.repeat()  # for val to
    dataset = dataset.map(
        lambda x: datasets_map[dataset_name].parse_single_exmp(x, is_training=is_training, label_num=label_num))
    if is_shuffle:
        dataset = dataset.shuffle(shuffle_num)            # shuffle
    dataset = dataset.batch(batch_size)
    return dataset

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

def get_dataset(name,
                dataset_dir,
                split_name,
                batch_size=32,
                resize_height=224,
                resize_width=224,
                label_num = 2,
                file_pattern=None,
                reader=None):


    """Given a dataset name and a split_name returns a Dataset.

    Args:
    name: String, the name of the dataset.
    split_name: A train/test split name.
    dataset_dir: The directory where the dataset files are stored.
    file_pattern: The file pattern to use for matching the dataset source files.
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
    reader defined by each dataset is used.

    Returns:
    A `Dataset`
    dataNums

    Raises:
        ValueError: If the dataset `name` is unknown.
    """
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    file_names = [os.path.join(dataset_dir, i) for i in os.listdir(dataset_dir)]
    print(file_names)
    if len(file_names) == 0:
        raise ValueError("dataSet %s dir has no files" % dataset_dir)
    if split_name is 'train':
        datasets = create_dataset(
            file_names, batch_size, is_training=True,
            is_shuffle=True, label_num=label_num, dataset_name=name)
    elif split_name is 'val':
        datasets = create_dataset(
            file_names, batch_size, is_training=False, n_repeats=-1, label_num=label_num, dataset_name=name)
    else:
        #test_set
        datasets = create_dataset(
            file_names, batch_size, is_training=False, n_repeats=1, is_shuffle=False, label_num=label_num, dataset_name=name)

    return datasets
