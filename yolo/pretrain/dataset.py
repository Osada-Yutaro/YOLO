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

        print(self.DATA_DIR)

    def load_train(self, start_index, end_index):
        import os
        import numpy as np
        import cv2
        eye = np.eye(1000)
        x_data = np.array(list(map(
            lambda s: cv2.resize(cv2.imread(os.path.join(self.DATA_DIR, 'Images', s) + '.png'), (224, 224)),
            self.TRAIN_LIST[start_index:end_index])))
        y_data = np.array(list(map(
            lambda s: eye[int(s.split('_')[0])],
            self.TRAIN_LIST[start_index:end_index])))
        return x_data, y_data

    def load_validation(self, start_index, end_index):
        import os
        import numpy as np
        import cv2
        eye = np.eye(1000)
        x_data = np.array(list(map(
            lambda s: cv2.resize(cv2.imread(os.path.join(self.DATA_DIR, 'Images', s) + '.png'), (224, 224)),
            self.VALIDATION_LIST[start_index:end_index])))
        y_data = np.array(list(map(
            lambda s: eye[int(s.split('_')[0])],
            self.VALIDATION_LIST[start_index:end_index])))
        return x_data, y_data

    def shuffle(self):
        import random as rnd
        rnd.shuffle(self.TRAIN_LIST)
