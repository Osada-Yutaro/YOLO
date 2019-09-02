class __Data:
    from abc import ABCMeta, abstractmethod
    __metaclass__ = ABCMeta
    def __init__(self, data_dir):
        import os
        f_tr = open(os.path.join(data_dir, 'train.txt'), 'r')
        f_vl = open(os.path.join(data_dir, 'validation.txt'), 'r')
        self.TRAIN_LIST = f_tr.read().split('\n')[0:-1]
        self.VALIDATION_LIST = f_vl.read().split('\n')[0:-1]
        self.TRAIN_DATA_SIZE = len(self.TRAIN_LIST)
        self.VALIDATION_DATA_SIZE = len(self.VALIDATION_LIST)
        self.DATA_DIR = data_dir
        f_tr.close()
        f_vl.close()

    @abstractmethod
    def load_train(self, start_index, end_index, prepro=False):
        pass

    @abstractmethod
    def load_validation(self, start_index, end_index):
        pass

    def shuffle(self):
        import random as rnd
        rnd.shuffle(self.TRAIN_LIST)
