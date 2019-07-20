class Data:
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

    def load_train(self, start_index, end_index, prepro):
        from . import preprocessing as pp
        import numpy as np
        x_data = np.array(list(map(lambda fn: pp.convert_X(self.DATA_DIR, fn), self.TRAIN_LIST[start_index:end_index])))
        y_data = np.array(list(map(lambda fn: pp.convert_Y(self.DATA_DIR, fn), self.TRAIN_LIST[start_index:end_index])))
        if prepro:
            for i in range(end_index - start_index):
                x_data[i], y_data[i] = pp.random_shift(x_data[i], y_data[i])
                x_data[i], y_data[i] = pp.random_reverse(x_data[i], y_data[i])
        return x_data, y_data

    def load_validation(self, start_index, end_index):
        from . import preprocessing as pp
        import numpy as np
        x_data = np.array(list(map(lambda fn: pp.convert_X(self.DATA_DIR, fn), self.VALIDATION_LIST[start_index:end_index])))
        y_data = np.array(list(map(lambda fn: pp.convert_Y(self.DATA_DIR, fn), self.VALIDATION_LIST[start_index:end_index])))
        return x_data, y_data

    def shuffle(self):
        import random as rnd
        rnd.shuffle(self.TRAIN_LIST)
