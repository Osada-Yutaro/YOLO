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
        eps = 1e-4
        return tf.nn.batch_normalization(x, mean, variance, offset, scale, eps)

def batch_normalized_conv2d(x, shape, step=1, name=None):
    import tensorflow as tf
    conv = conv2d(x, shape, step, name)
    return batch_normalization(conv2d(x, shape, step, name), conv.shape[1:], name)

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
        return tf.nn.relu(tf.reshape(tf.matmul(x, w), [-1, 7, 7, 30]))

    return __eighth_block(__seventh_block(__sixth_block(__fifth_block(__fourth_block(__third_block(__second_block(__first_block(x)))))), keep_prob))

LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5
BATCH_SIZE = 16
DECAY = 0.0005
TRAIN_DATA_SIZE = 0
VALIDATION_DATA_SIZE = 0

parsed_xml_list_train = None
parsed_xml_list_validation = None

def convert_X(directory, parsed_xml):
    import numpy as np
    import cv2
    return cv2.resize(cv2.imread(directory + 'JPEGImages/' + parsed_xml.find('filename').text), dsize=(448, 448))

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
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= random.random()/2 + 1.0
    hsv[:, :, 2] *= random.random()/2 + 1.0
    hsv = np.minimum(hsv, 255.).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def random_shift(inp, oup):
    import random
    import numpy as np
    import cv2

    width = 448
    height = 448
    x_min = int(width*random.random()/10)
    x_max = int(width*(random.random()/10 + 0.9))
    y_min = int(height*random.random()/10)
    y_max = int(height*(random.random()/10 + 0.9))

    W_new = x_max - x_min
    H_new = y_max - y_min

    inp_new = cv2.resize(inp[x_min:x_max, y_min:y_max, :], dsize=(448, 448))
    oup_new = np.zeros([7, 7, 30], dtype='float32')

    for sx in range(S):
        for sy in range(S):
            for b in range(B):
                boundingbox = oup[sx, sy, C + 5*b: C + 5*b + 5]
                x_mid = (sx + boundingbox[0])*width/S
                y_mid = (sy + boundingbox[1])*height/S

                left = x_mid - boundingbox[2]*width/2
                right = x_mid + boundingbox[2]*width/2
                up = y_mid - boundingbox[3]*height/2
                down = y_mid + boundingbox[3]*height/2

                if (boundingbox[4] == 1.) and (x_min < right < x_max or x_min < left < x_max) and (y_min < down < y_max or y_min < up < y_max):
                    l_new = max(left, x_min)
                    r_new = min(right, x_max)
                    u_new = max(up, y_min)
                    d_new = min(down, y_max)

                    x_mid_new = ((l_new + r_new)/2 - x_min)*S/W_new
                    y_mid_new = ((d_new + u_new)/2 - y_min)*S/H_new
                    w_new = (r_new - l_new)/W_new
                    h_new = (d_new - u_new)/H_new

                    sx_new = int(x_mid_new)
                    sy_new = int(y_mid_new)

                    oup_new[sx_new, sy_new, C + 5*b + 0] = x_mid_new - sx_new
                    oup_new[sx_new, sy_new, C + 5*b + 1] = y_mid_new - sy_new
                    oup_new[sx_new, sy_new, C + 5*b + 2] = w_new
                    oup_new[sx_new, sy_new, C + 5*b + 3] = h_new
                    oup_new[sx_new, sy_new, C + 5*b + 4] = 1.

                    oup_new[sx_new, sy_new, 0:C] = np.maximum(oup[sx, sy, 0:C], oup_new[sx_new, sy_new, 0:C])

    return inp_new, oup_new

def load_dataset(directory):
    import xml.etree.ElementTree as ET
    import glob

    global parsed_xml_list_train
    global parsed_xml_list_validation
    global TRAIN_DATA_SIZE
    global VALIDATION_DATA_SIZE
    if directory[-1] != '/':
        directory = directory + '/'
    parsed_xml_list_train = list(map(ET.parse, glob.glob(directory + 'Annotations/Train/*')))
    parsed_xml_list_validation = list(map(ET.parse, glob.glob(directory + 'Annotations/Validation/*')))
    TRAIN_DATA_SIZE = len(parsed_xml_list_train)
    VALIDATION_DATA_SIZE = len(parsed_xml_list_validation)

def load_train(directory, start_index, end_index):
    import numpy as np
    global parsed_xml_list_train
    if parsed_xml_list_train is None:
        load_dataset(directory)
    x_data = np.array(list(map(lambda xml: random_adjust(convert_X(directory, xml)), parsed_xml_list_train[start_index:end_index])))
    y_data = np.array(list(map(convert_Y, parsed_xml_list_train[start_index:end_index])))

    for i in range(end_index - start_index):
        x_data[i], y_data[i] = random_shift(x_data[i], y_data[i])
    return x_data, y_data

def load_validation(directory, start_index, end_index):
    import numpy as np
    global parsed_xml_list_validation
    if parsed_xml_list_validation is None:
        load_dataset(directory)
    x_data = np.array(list(map(lambda xml: convert_X(directory, xml), parsed_xml_list_validation[start_index:end_index])))
    y_data = np.array(list(map(convert_Y, parsed_xml_list_validation[start_index:end_index])))
    return x_data.astype(np.float32), y_data

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

    eps = 1e-8*tf.ones([D, S, S, B])
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

    err_d = loss_d(y, y_pred, D)/tf.cast(D, tf.float32)
    err_w = DECAY*loss_w()
    minimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(err_d + err_w)

    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()

    load_dataset(res_dir)

    ckpt = tf.train.get_checkpoint_state(model_dir)
    with tf.Session() as sess:
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, model_dir + 'weights.ckpt')
        else:
            sess.run(init)
        print('epoch, training error, validation error, weight error')
        for epoch in range(start_epoch, start_epoch + epoch_size):
            random.shuffle(parsed_xml_list_train)

            count_train = 0
            while count_train < TRAIN_DATA_SIZE:
                nextcount = min(count_train + BATCH_SIZE, TRAIN_DATA_SIZE)
                x_train, y_train = load_train(res_dir, count_train, nextcount)
                sess.run(minimize, feed_dict={x: x_train, y: y_train, D: nextcount - count_train, keep_prob: .5, learning_rate: lr})
                count_train = nextcount

            if epoch%1 == 0:
                count_train = 0
                err_train = 0
                while count_train < TRAIN_DATA_SIZE:
                    nextcount = min(count_train + BATCH_SIZE, TRAIN_DATA_SIZE)
                    x_train, y_train = load_train(res_dir, count_train, nextcount)
                    err_train += sess.run(tf.cast(D, tf.float32)*err_d/TRAIN_DATA_SIZE, feed_dict={x: x_train, y: y_train, D: nextcount - count_train, keep_prob: 1.})
                    count_train = nextcount

                count_validation = 0
                err_validation = 0
                while count_validation < VALIDATION_DATA_SIZE:
                    nextcount = min(count_validation + BATCH_SIZE, VALIDATION_DATA_SIZE)
                    x_validation, y_validation = load_validation(res_dir, count_validation, nextcount)
                    err_validation += sess.run(tf.cast(D, tf.float32)*err_d/VALIDATION_DATA_SIZE, feed_dict={x: x_validation, y: y_validation, D: nextcount - count_validation, keep_prob: 1.})
                    count_validation = nextcount

                print(epoch, err_train, err_validation, sess.run(err_w))
        saver.save(sess, model_dir + 'weights.ckpt')
