S = 7
B = 2
C = 20
INDICES = {'diningtable': 0, 'chair': 1, 'horse': 2, 'person': 3, 'tvmonitor': 4, 'bird': 5, 'cow': 6, 'dog': 7, 'bottle': 8, 'pottedplant': 9, 'aeroplane': 10, 'car': 11, 'cat': 12, 'sheep': 13, 'bicycle': 14, 'sofa': 15, 'boat': 16, 'train': 17, 'motorbike': 18, 'bus': 19}
REVERSE_RESOLUTION = {0: 'diningtable', 1: 'chair', 2: 'horse', 3: 'person', 4: 'tvmonitor', 5: 'bird', 6: 'cow', 7: 'dog', 8: 'bottle', 9: 'pottedplant', 10: 'aeroplane', 11: 'car', 12: 'cat', 13: 'sheep', 14: 'bicycle', 15: 'sofa', 16: 'boat', 17: 'train', 18: 'motorbike', 19: 'bus'}

def weight_variable(shape, name):
    import tensorflow as tf
    with tf.variable_scope('yolo', reuse=tf.AUTO_REUSE):
        w = tf.get_variable(name, initializer=tf.truncated_normal(shape, stddev=0.03))
        return w

def conv2d(x, shape, step=1, name=None):
    import tensorflow as tf
    return tf.nn.conv2d(x, weight_variable(shape, name), [1, step, step, 1], 'SAME')

def leaky_relu(x):
    import tensorflow as tf
    return tf.nn.leaky_relu(x, alpha=0.1)

def max_pool(x, r, step):
    import tensorflow as tf
    return tf.nn.max_pool(x, [1, r, r, 1], [1, step, step, 1], 'SAME')

def batch_normalization(x, shape, name):
    import tensorflow as tf
    with tf.variable_scope('yolo', reuse=tf.AUTO_REUSE):
        mean = tf.reduce_mean(x, axis=0)
        variance = tf.reduce_mean(tf.square(x - mean))
        offset = tf.get_variable(name + '_offset', initializer=tf.zeros(shape))
        scale = tf.get_variable(name + '_scale', initializer=tf.ones(shape))
        eps = 1e-2
        return tf.nn.batch_normalization(x, mean, variance, offset, scale, eps)

def batch_normalized_conv2d(x, shape, step=1, name=None):
    import tensorflow as tf
    conv = conv2d(x, shape, step, name)
    return batch_normalization(conv2d(x, shape, step, name), [conv.shape[1], conv.shape[2], conv.shape[3]], name)

def model(x, keep_prob=1.):
    import tensorflow as tf
    def __first_block(x):
        return max_pool(leaky_relu(conv2d(x, [7, 7, 3, 192], 2, name='w_1')), 2, 2)
    def __second_block(x):
        return max_pool(leaky_relu(conv2d(x, [3, 3, 192, 128], 1, name='w_2')), 2, 2)
    def __third_block(x):
        layer_1 = leaky_relu(conv2d(x, [1, 1, 128, 256], name='w_3_1'))
        layer_2 = leaky_relu(conv2d(layer_1, [3, 3, 256, 256], name='w_3_2'))
        layer_3 = leaky_relu(conv2d(layer_2, [1, 1, 256, 512], name='w_3_3'))
        layer_4 = leaky_relu(conv2d(layer_3, [3, 3, 512, 256], name='w_3_4'))
        return max_pool(layer_4, 2, 2)
    def __fourth_block(x):
        layer_1 = leaky_relu(conv2d(x, [1, 1, 256, 512], name='w_4_1'))
        layer_2 = leaky_relu(conv2d(layer_1, [3, 3, 512, 256], name='w_4_2'))
        layer_3 = leaky_relu(conv2d(layer_2, [1, 1, 256, 512], name='w_4_3'))
        layer_4 = leaky_relu(conv2d(layer_3, [3, 3, 512, 256], name='w_4_4'))
        layer_5 = leaky_relu(conv2d(layer_4, [1, 1, 256, 512], name='w_4_5'))
        layer_6 = leaky_relu(conv2d(layer_5, [3, 3, 512, 256], name='w_4_6'))
        layer_7 = leaky_relu(conv2d(layer_6, [1, 1, 256, 512], name='w_4_7'))
        layer_8 = leaky_relu(conv2d(layer_7, [3, 3, 512, 512], name='w_4_8'))
        layer_9 = leaky_relu(conv2d(layer_8, [1, 1, 512, 1024], name='w_4_9'))
        layer_10 = leaky_relu(conv2d(layer_9, [3, 3, 1024, 512], name='w_4_10'))
        return max_pool(layer_10, 2, 2)
    def __fifth_block(x):
        layer_1 = leaky_relu(conv2d(x, [1, 1, 512, 1024], name='w_5_1'))
        layer_2 = leaky_relu(conv2d(layer_1, [3, 3, 1024, 512], name='w_5_2'))
        layer_3 = leaky_relu(conv2d(layer_2, [1, 1, 512, 1024], name='w_5_3'))
        layer_4 = leaky_relu(conv2d(layer_3, [3, 3, 1024, 1024], name='w_5_4'))
        layer_5 = leaky_relu(conv2d(layer_4, [3, 3, 1024, 1024], name='w_5_5'))
        return leaky_relu(conv2d(layer_5, [3, 3, 1024, 1024], 2, name='w_5_6'))
    def __sixth_block(x):
        layer_1 = leaky_relu(conv2d(x, [3, 3, 1024, 1024], name='w_6_1'))
        return leaky_relu(conv2d(layer_1, [3, 3, 1024, 1024], name='w_6_2'))
    def __seventh_block(x, keep_prob):
        w = weight_variable([7*7*1024, 4096], name='w_7')
        conn = leaky_relu(tf.matmul(tf.reshape(x, [-1, 7*7*1024]), w))
        return tf.nn.dropout(conn, rate=1-keep_prob)
    def __eighth_block(x):
        w = weight_variable([4096, 7*7*30], name='w_8')
        return tf.reshape(tf.matmul(x, w), [-1, 7, 7, 30])

    return __eighth_block(__seventh_block(__sixth_block(__fifth_block(__fourth_block(__third_block(__second_block(__first_block(x)))))), keep_prob))

LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5
BATCH_SIZE = 16
DECAY = 0.0005
DATA_SIZE = 5011

parsed_xml_list_train = None
parsed_xml_list_test = None

def convert_X(directory, parsed_xml):
    import numpy as np
    from PIL import Image
    return Image.open(directory + 'JPEGImages/' + parsed_xml.find('filename').text).resize((448, 448))

def convert_Y(parsed_xml):
    import numpy as np
    width = float(parsed_xml.find('size/width').text)
    height = float(parsed_xml.find('size/height').text)
    y_data = np.zeros((S, S, 5*B + C), dtype='float32')
    for sx in range(S):
        for sy in range(S):
            b = 0
            for obj in parsed_xml.findall('object'):
                if b == B:
                    break

                xmin = float(obj.find('bndbox/xmin').text)
                ymin = float(obj.find('bndbox/ymin').text)
                xmax = float(obj.find('bndbox/xmax').text)
                ymax = float(obj.find('bndbox/ymax').text)
                xmid = (xmin + xmax)/2
                ymid = (ymin + ymax)/2

                if sx*width/S <= xmid < (sx + 1)*width/S and sy*height/S <= ymid < (sy + 1)*height/S:
                    y_data[sx][sy][C + 5*b] = S*xmid/width - sx
                    y_data[sx][sy][C + 5*b + 1] = S*ymid/height - sy
                    y_data[sx][sy][C + 5*b + 2] = (xmax - xmin)/width
                    y_data[sx][sy][C + 5*b + 3] = (ymax - ymin)/height
                    y_data[sx][sy][C + 5*b + 4] = 1.

                    y_data[sx][sy][INDICES[obj.find('name').text]] = 1.
                    b += 1
    return y_data

def random_adjust(img):
    import random
    import numpy as np
    from PIL import Image
    from PIL import ImageEnhance
    return np.asarray(ImageEnhance.Color(ImageEnhance.Brightness(img).enhance(random.random() + 0.5)).enhance(random.random() + 0.5), dtype='float32')

def random_shift(x, y):
    width, height = x.shape[0], x.shape[1]

def load_dataset(directory):
    import xml.etree.ElementTree as ET
    import glob

    global parsed_xml_list_train
    global parsed_xml_list_test
    if directory[-1] != '/':
        directory = directory + '/'
    parsed_xml_list_train = list(map(ET.parse, glob.glob(directory + 'Annotations/Train/*')))
    parsed_xml_list_test = list(map(ET.parse, glob.glob(directory + 'Annotations/Test/*')))
    DATA_SIZE = len(parsed_xml_list_train) + len(parsed_xml_list_test)

def load_train(directory, start_index, end_index):
    import numpy as np
    global parsed_xml_list_train
    if parsed_xml_list_train is None:
        load_dataset(directory)
    x_data = np.array(list(map(lambda xml: random_adjust(convert_X(directory, xml)), parsed_xml_list_train[start_index:end_index])))
    y_data = np.array(list(map(convert_Y, parsed_xml_list_train[start_index:end_index])))
    return x_data, y_data

def load_test(directory, start_index, end_index):
    import numpy as np
    global parsed_xml_list_test
    if parsed_xml_list_test is None:
        load_dataset(directory)
    x_data = np.array(list(map(lambda xml: np.asarray(convert_X(directory, xml), dtype='float32'), parsed_xml_list_test[start_index:end_index])), dtype='float32')
    y_data = np.array(list(map(convert_Y, parsed_xml_list_test[start_index:end_index])))
    return x_data, y_data

def loss_w():
    import tensorflow as tf
    with tf.variable_scope('yolo', reuse=True):
        err = tf.reduce_sum(tf.square(tf.get_variable('w_1')))
        err += tf.reduce_sum(tf.square(tf.get_variable('w_2')))
        for i in range(1, 5):
            err += tf.reduce_sum(tf.square(tf.get_variable('w_3_' + str(i))))
        for i in range(1, 11):
            err += tf.reduce_sum(tf.square(tf.get_variable('w_4_' + str(i))))
        for i in range(1, 7):
            err += tf.reduce_sum(tf.square(tf.get_variable('w_5_' + str(i))))
        for i in range(1, 3):
            err += tf.reduce_sum(tf.square(tf.get_variable('w_6_' + str(i))))
        err += tf.reduce_sum(tf.square(tf.get_variable('w_7')))
        err += tf.reduce_sum(tf.square(tf.get_variable('w_8')))
        return err

def position_loss(trgt, pred, D):
    import tensorflow as tf
    x_trgt = trgt[:, :, :, C:C + 5*B:5]
    y_trgt = trgt[:, :, :, C + 1:C + 5*B:5]
    confi_trgt = trgt[:, :, :, C + 4:C + 5*B:5]

    x_pred = pred[:, :, :, C:C + 5*B:5]
    y_pred = pred[:, :, :, C + 1:C + 5*B:5]

    x_loss = LAMBDA_COORD*tf.reduce_sum(tf.square(x_trgt - x_pred)*confi_trgt)
    y_loss = LAMBDA_COORD*tf.reduce_sum(tf.square(y_trgt - y_pred)*confi_trgt)
    return x_loss + y_loss

def size_loss(trgt, pred, D):
    import tensorflow as tf
    w_trgt = trgt[:, :, :, C + 2:C + 5*B:5]
    h_trgt = trgt[:, :, :, C + 3:C + 5*B:5]
    confi_trgt = trgt[:, :, :, C + 4:C + 5*B:5]

    w_pred = tf.nn.relu(pred[:, :, :, C + 2:C + 5*B:5])
    h_pred = tf.nn.relu(pred[:, :, :, C + 3:C + 5*B:5])

    eps = 1e-4*tf.ones([D, S, S, B])
    w_loss = LAMBDA_COORD*tf.reduce_sum(tf.square(tf.sqrt(w_trgt + eps) - tf.sqrt(w_pred + eps))*confi_trgt)
    h_loss = LAMBDA_COORD*tf.reduce_sum(tf.square(tf.sqrt(h_trgt + eps) - tf.sqrt(h_pred + eps))*confi_trgt)
    return w_loss + h_loss

def confidence_loss(trgt, pred, D):
    import tensorflow as tf
    confi_trgt = trgt[:, :, :, C + 4:C + 5*B:5]

    confi_pred = pred[:, :, :, C + 4:C + 5*B:5]

    confi_loss_obj = tf.reduce_sum(tf.square(confi_trgt - confi_pred)*confi_trgt)
    confi_loss_noobj = LAMBDA_NOOBJ*tf.reduce_sum(tf.square(confi_trgt - confi_pred)*(tf.ones([D, S, S, B]) - confi_trgt))

    return confi_loss_obj + confi_loss_noobj

def class_loss(trgt, pred, D):
    import tensorflow as tf
    p_trgt = trgt[:, :, :, 0:C]
    p_pred = pred[:, :, :, 0:C]
    pred_loss = tf.reduce_sum(tf.square(p_trgt - p_pred)*tf.reshape(tf.reduce_max(p_trgt, axis=[3]), [D, S, S, 1]))
    return pred_loss

def loss_d(trgt, pred, D):
    return position_loss(trgt, pred, D) + size_loss(trgt, pred, D) + confidence_loss(trgt, pred, D) + class_loss(trgt, pred, D)

def train(res_dir, model_dir, epoch_size=100, lr=1e-3, start_epoch=1):
    import random
    import tensorflow as tf

    if model_dir[-1] != '/':
        model_dir = model_dir + '/'

    x = tf.placeholder(tf.float32, [None, 448, 448, 3])
    y = tf.placeholder(tf.float32, [None, S, S, 5*B + C])
    D = tf.placeholder(tf.int32)
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    ckpt = tf.train.get_checkpoint_state(model_dir)
    y_pred = model(x, keep_prob)

    err_d = loss_d(y, y_pred, D)
    err_w = DECAY*loss_w()
    minimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(err_d/tf.cast(D, tf.float32) + err_w)

    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()

    load_dataset(res_dir)

    train_data_size = len(parsed_xml_list_train)
    test_data_size = len(parsed_xml_list_test)


    ckpt = tf.train.get_checkpoint_state(model_dir)
    with tf.Session() as sess:
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, model_dir + 'weights.ckpt')
        else:
            sess.run(init)
        count_train = 0
        err_train = 0
        while count_train < train_data_size:
            nextcount = min(count_train + BATCH_SIZE, train_data_size)
            x_train, y_train = load_train(res_dir, count_train, nextcount)
            err_train += sess.run(err_d/train_data_size, feed_dict={x: x_train, y: y_train, D: nextcount - count_train, keep_prob: 1.})
            count_train = nextcount

        count_test = 0
        err_test = 0
        while count_test < test_data_size:
            nextcount = min(count_test + BATCH_SIZE, test_data_size)
            x_test, y_test = load_test(res_dir, count_test, nextcount)
            err_test += sess.run(err_d/test_data_size, feed_dict={x: x_test, y: y_test, D: nextcount - count_test, keep_prob: 1.})
            count_test = nextcount
        print(0, err_train, err_test, sess.run(err_w))

        print('epoch, training error, test error, weight error')
        for epoch in range(start_epoch, start_epoch + epoch_size):
            random.shuffle(parsed_xml_list_train)

            count_train = 0
            while count_train < train_data_size:
                nextcount = min(count_train + BATCH_SIZE, train_data_size)
                x_train, y_train = load_train(res_dir, count_train, nextcount)
                sess.run(minimize, feed_dict={x: x_train, y: y_train, D: nextcount - count_train, keep_prob: 1., learning_rate: lr})
                count_train = nextcount

            if epoch%1 == 0:
                count_train = 0
                err_train = 0
                while count_train < train_data_size:
                    nextcount = min(count_train + BATCH_SIZE, train_data_size)
                    x_train, y_train = load_train(res_dir, count_train, nextcount)
                    err_train += sess.run(err_d/train_data_size, feed_dict={x: x_train, y: y_train, D: nextcount - count_train, keep_prob: 1.})
                    count_train = nextcount

                count_test = 0
                err_test = 0
                while count_test < test_data_size:
                    nextcount = min(count_test + BATCH_SIZE, test_data_size)
                    x_test, y_test = load_test(res_dir, count_test, nextcount)
                    err_test += sess.run(err_d/test_data_size, feed_dict={x: x_test, y: y_test, D: nextcount - count_test, keep_prob: 1.})
                    count_test = nextcount

                print(epoch, err_train, err_test, sess.run(err_w))
        saver.save(sess, model_dir + 'weights.ckpt')
