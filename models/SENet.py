import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from  tensorflow.contrib.layers import  batch_norm, flatten
from tensorflow.contrib.framework import  arg_scope
import numpy as np

class SE_CNN():
    def __init__(self, class_num):
        self.class_num = class_num

