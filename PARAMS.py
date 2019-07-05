"""
     -*- coding: utf-8 -*-
    @Project: PyCharm
    @File    : PARAMS.py
    @Author  : LLL
    @Site    : 
    @Email   : lilanluo@stu.xmu.edu.cn
    @Date    : 2019/5/13 11:03
    @info   :
"""
import os

class Params:
    params = None

    def __init__(self):
        self.params = dict()
        self.params['path'] = dict()
        self.params['model'] = dict()
        # file_params
        self.params['path']['train_rex'] = "dataset/iwatch_224_record/train/train-*"
        self.params['path']['test_rex'] = "dataset/iwatch_224_record/test/test-*"
        self.params['path']['val_rex'] = "dataset/iwatch_224_record/val/val-*"
        self.params['path']['model_path'] = 'shell_model'
        self.params['path']['train_tensorBoardPath'] = 'tensorboard/train'
        self.params['path']['test_tensorBoardPath'] = 'tensorboard/test'
        # model_params
        self.params['model']['batchsize'] = 32  # the batchsize
        self.params['model']['baseLR'] = 0.001 #the learning rate, initial one0.0001
        self.params['model']['decayLR'] = 0.95
        self.params['model']['classNum'] = 2
        self.params['model']['move_avg_decay'] = 0.95
        self.params['model']['width'] = 224
        self.params['model']['height'] = 224
        self.params['model']['depth'] = 1
        self.params['model']['keep_prob'] = 0.9
        self.check_params()
    def check_params(self):
        dir_list = [self.params['path']['train_tensorBoardPath'],
                    self.params['path']['test_tensorBoardPath'],
                    self.params['path']['model_path']]  #获取父级别目录
        for dir in dir_list:
            if not os.path.exists(dir):
                os.makedirs(dir)
