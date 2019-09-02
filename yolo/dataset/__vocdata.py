from yolo.dataset.__data import __Data

class VOCData(__Data):
    def load_train(self, start_index, end_index, prepro=False):
        from yolo.train import preprocessing as pp
        import numpy as np
        x_data = np.array(list(map(
            lambda fn: pp.convert_X(self.DATA_DIR, fn),
            self.TRAIN_LIST[start_index:end_index])))
        y_data = np.array(list(map(
            lambda fn: pp.convert_Y(self.DATA_DIR, fn),
            self.TRAIN_LIST[start_index:end_index])))
        if prepro:
            for i in range(end_index - start_index):
                x_data[i], y_data[i] = pp.random_shift(x_data[i], y_data[i])
                x_data[i], y_data[i] = pp.random_reverse(x_data[i], y_data[i])
        return x_data, y_data

    def load_validation(self, start_index, end_index):
        from yolo.train import preprocessing as pp
        import numpy as np
        x_data = np.array(list(map(
            lambda fn: pp.convert_X(self.DATA_DIR, fn),
            self.VALIDATION_LIST[start_index:end_index])))
        y_data = np.array(list(map(
            lambda fn: pp.convert_Y(self.DATA_DIR, fn),
            self.VALIDATION_LIST[start_index:end_index])))
        return x_data, y_data
