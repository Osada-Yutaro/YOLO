import xml.etree.ElementTree as ET
import glob
import sys

import tensorflow as tf
import numpy as np
from PIL import Image
import yolo

S = 7
B = 2
C = 20
INDICES = {'diningtable': 0, 'chair': 1, 'horse': 2, 'person': 3, 'tvmonitor': 4, 'bird': 5, 'cow': 6, 'dog': 7, 'bottle': 8, 'pottedplant': 9, 'aeroplane': 10, 'car': 11, 'cat': 12, 'sheep': 13, 'bicycle': 14, 'sofa': 15, 'boat': 16, 'train': 17, 'motorbike': 18, 'bus': 19}
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5
BATCH_SIZE = 32
DECAY = 0.0005
DATA_SIZE = 5011

parsed_xml_list = None

def convert_X(directory, parsed_xml):
    return np.array(Image.open(directory + 'JPEGImages/' + parsed_xml.find('filename').text).resize((448, 448)))

def convert_Y(parsed_xml):
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

                if 0 <= (xmax + xmin)/2 - sx*width/S < width/S and 0 <= (ymax + ymin)/2 - sy*height/S < height/S:
                    y_data[sx][sy][5*b] = S*(xmax + xmin)/2/width - sx
                    y_data[sx][sy][5*b + 1] = S*(ymax + ymin)/2/height - sy
                    y_data[sx][sy][5*b + 2] = (xmax - xmin)/width
                    y_data[sx][sy][5*b + 3] = (ymax - ymin)/height
                    y_data[sx][sy][5*b + 4] = 1.

                    b += 1
    return y_data

def load_dataset(directory, start_index, end_index):
    global parsed_xml_list
    if parsed_xml_list is None:
        if directory[-1] != '/':
            directory = directory + '/'
        parsed_xml_list = list(map(ET.parse, glob.glob(directory + 'Annotations/*')))
    x_data = np.array(list(map(lambda xml: convert_X(directory, xml), parsed_xml_list[start_index:end_index])))
    y_data = np.array(list(map(convert_Y, parsed_xml_list[start_index:end_index])))
    return x_data, y_data

def loss_w():
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

def loss_5(output_target, output_pred):
    # 幅と高さの損失の微分係数の計算で0除算が起きないようにしている
    def size_loss_true_fn():
        eps = 1e-4
        return LAMBDA_COORD*(tf.square(tf.sqrt(output_target[2] + eps) - tf.sqrt(output_pred[2] + eps)) + tf.square(tf.sqrt(output_target[3] + eps) - tf.sqrt(output_pred[3] + eps)))
    def size_loss_false_fn():
        return 0.
    pos_loss = LAMBDA_COORD*output_target[4]*(tf.square(output_target[0] - output_pred[0]) + tf.square(output_target[1] - output_pred[1]))
    size_loss = tf.cond(output_target[4] > 0.99, size_loss_true_fn, size_loss_false_fn)
    confi_loss = output_target[4]*tf.square(output_target[4] - output_pred[4])
    return pos_loss + size_loss + confi_loss

def loss_4(output_target, output_pred):
    return tf.reduce_sum(tf.square(output_target - output_pred))

def loss_3(output_target, output_pred):
    def upd(x, y):
        return x + 1, y + loss_4(output_target[5*x:5*(x + 1)], output_pred[5*x:5*(x + 1)])
    return tf.while_loop(lambda x, y: x < B, upd, (0, 0.))[1] + loss_5(output_target[5*B + 1:5*B + C], output_pred[5*B + 1:5*B + C])

def loss_2(output_target, output_pred):
    def upd(x, y):
        return x + 1, y + loss_3(output_target[x], output_pred[x])
    return tf.while_loop(lambda x, y: x < S, upd, (0, 0.))[1]

def loss_1(output_target, output_pred):
    def upd(x, y):
        return x + 1, y + loss_2(output_target[x], output_pred[x])
    return tf.while_loop(lambda x, y: x < S, upd, (0, 0.))[1]

def loss_d(output_target, output_pred, D):
    def upd(x, y):
        return x + 1, y + loss_1(output_target[x], output_pred[x])
    return tf.while_loop(lambda x, y: x < D, upd, (0, 0.))[1]

def main():
    args = sys.args
    res_dir = args[1]
    model_dir = args[2]
    if model_dir[-1] != '/':
        model_dir = model_dir + '/'

    train_data_size = int(DATA_SIZE*0.75)
    test_data_size = DATA_SIZE - train_data_size

    x = tf.placeholder(tf.float32, [None, 448, 448, 3])
    y = tf.placeholder(tf.float32, [None, S, S, 5*B + C])
    D = tf.placeholder(tf.int32)
    keep_prob = tf.placeholder(tf.float32)

    y_pred = yolo.model(x, keep_prob)

    err = (loss_d(y, y_pred, D) + DECAY*loss_w())/tf.cast(D, tf.float32)
    train1 = tf.train.GradientDescentOptimizer(1e-2).minimize(err)
    train2 = tf.train.GradientDescentOptimizer(1e-3).minimize(err)
    train3 = tf.train.GradientDescentOptimizer(1e-4).minimize(err)

    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print('epoch, training error, test error, weight error')
        for epoch in range(1, 136):
            count_train = 0
            while count_train < train_data_size:
                nextcount = min(count_train + BATCH_SIZE, train_data_size)
                x_train, y_train = load_dataset(res_dir, count_train, nextcount)
                if epoch < 76:
                    sess.run(train1, feed_dict={x: x_train, y: y_train, D: nextcount - count_train, keep_prob: .5})
                elif epoch < 106:
                    sess.run(train2, feed_dict={x: x_train, y: y_train, D: nextcount - count_train, keep_prob: .5})
                else:
                    sess.run(train3, feed_dict={x: x_train, y: y_train, D: nextcount - count_train, keep_prob: .5})
                count_train = nextcount
            if epoch%5 == 0:
                count_train = 0
                err_train = 0
                while count_train < train_data_size:
                    nextcount = min(count_train + BATCH_SIZE, train_data_size)
                    x_train, y_train = load_dataset(res_dir, count_train, nextcount)
                    err_train += sess.run(loss_d(y, y_pred, D), feed_dict={x: x_train, y: y_train, D: nextcount - count_train, keep_prob: 1.})
                    count_train = nextcount

                count_test = 0
                err_test = 0
                while count_test < test_data_size:
                    nextcount = min(count_test + BATCH_SIZE, test_data_size)
                    x_test, y_test = load_dataset(res_dir, count_train + count_test, count_train + nextcount)
                    err_test += sess.run(loss_d(y, y_pred, D), feed_dict={x: x_test, y: y_test, D: nextcount - count_test, keep_prob: 1.})
                    count_test = nextcount

                err_w = sess.run(loss_w())
                print(epoch, err_train, err_test, err_w)
        saver.save(sess, model_dir + 'weights.ckpt')

main()
