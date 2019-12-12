from yolo.dataset.__data import __Data

class ImageNetData(__Data):
    def load_train(self, start_index, end_index, prepro=False):
        import os
        import numpy as np
        import cv2
        from yolo.train import preprocessing as pp
        eye = np.eye(1000)
        x_data = np.array(list(map(
            lambda s: cv2.resize(
                cv2.imread(os.path.join(self.DATA_DIR, 'Images', s) + '.png'),
                (224, 224)),
            self.TRAIN_LIST[start_index:end_index])))
        y_data = np.array(list(map(
            lambda s: eye[int(s.split('_')[0])],
            self.TRAIN_LIST[start_index:end_index])))
        if prepro:
            for i in range(end_index - start_index):
                x_data[i], y_data[i] = pp.random_shift(x_data[i], y_data[i])
                x_data[i], y_data[i] = pp.random_reverse(x_data[i], y_data[i])
        return x_data, y_data

    def load_validation(self, start_index, end_index):
        import os
        import numpy as np
        import cv2
        eye = np.eye(1000)
        x_data = np.array(list(map(
            lambda s: cv2.resize(
                cv2.imread(os.path.join(self.DATA_DIR, 'Images', s) + '.png'),
                (224, 224)
            ),
            self.VALIDATION_LIST[start_index:end_index])))
        y_data = np.array(list(map(
            lambda s: eye[int(s.split('_')[0])],
            self.VALIDATION_LIST[start_index:end_index])))
        return x_data, y_data
