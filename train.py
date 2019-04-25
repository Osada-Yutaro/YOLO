import xml.etree.ElementTree as ET
import glob
import sys
import random

import tensorflow as tf
import numpy as np
from PIL import Image
import yolo

S = yolo.S
B = yolo.B
C = yolo.C
INDICES = yolo.INDICES
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5
BATCH_SIZE = 16
DECAY = 0.0002
DATA_SIZE = 5011

parsed_xml_list_train = None
parsed_xml_list_test = None

def convert_X(directory, parsed_xml):
    return np.array(Image.open(directory + 'JPEGImages/' + parsed_xml.find('filename').text).resize((448, 448)), dtype='float32')

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

def load_dataset(directory):
    global parsed_xml_list_train
    global parsed_xml_list_test
    if directory[-1] != '/':
        directory = directory + '/'
    parsed_xml_list = list(map(ET.parse, glob.glob(directory + 'Annotations/*')))
    parsed_xml_list_train = parsed_xml_list[0:int(DATA_SIZE*0.75)]
    parsed_xml_list_test = parsed_xml_list[int(DATA_SIZE*0.75):DATA_SIZE]
    return

def load_train(directory, start_index, end_index):
    global parsed_xml_list_train
    if parsed_xml_list_train is None:
        load_dataset(directory)
    x_data = np.array(list(map(lambda xml: convert_X(directory, xml), parsed_xml_list_train[start_index:end_index])))
    y_data = np.array(list(map(convert_Y, parsed_xml_list_train[start_index:end_index])))
    return x_data, y_data

def load_test(directory, start_index, end_index):
    global parsed_xml_list_test
    if parsed_xml_list_test is None:
        load_dataset(directory)
    x_data = np.array(list(map(lambda xml: convert_X(directory, xml), parsed_xml_list_train[start_index:end_index])))
    y_data = np.array(list(map(convert_Y, parsed_xml_list_train[start_index:end_index])))
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

def loss_d(output_target, output_pred, D):
    p_target = output_target[:, :, :, 0:C]
    x_target = output_target[:, :, :, C:5*B + C: 5]
    y_target = output_target[:, :, :, C + 1:5*B + C: 5]
    w_target = output_target[:, :, :, C + 2:5*B + C: 5]
    h_target = output_target[:, :, :, C + 3:5*B + C: 5]
    confi_target = output_target[:, :, :, C + 4:5*B + C: 5]

    p_pred = output_pred[:, :, :, 0:C]
    x_pred = output_pred[:, :, :, C:5*B + C: 5]
    y_pred = output_pred[:, :, :, C + 1:5*B + C: 5]
    w_pred = output_pred[:, :, :, C + 2:5*B + C: 5]
    h_pred = output_pred[:, :, :, C + 3:5*B + C: 5]
    confi_pred = output_pred[:, :, :, C + 4:5*B + C: 5]

    pos_err = LAMBDA_COORD*tf.reduce_sum((tf.square(x_target - x_pred) + tf.square(y_target - y_pred))*confi_target)

    eps = 1e-8*tf.ones([D, S, S, B])
    size_err = tf.reduce_sum((tf.square(tf.sqrt(w_target + eps) - tf.sqrt(w_pred + eps)) + tf.square(tf.sqrt(h_target + eps) - tf.sqrt(h_pred + eps)))*confi_target)

    confi_err_obj = tf.reduce_sum(tf.square(confi_target - confi_pred)*confi_target)
    confi_err_noobj = LAMBDA_NOOBJ*tf.reduce_sum(tf.square(confi_target - confi_pred)*(tf.ones([D, S, S, B]) - confi_target))

    pred_err = tf.reduce_sum(tf.square(p_target - p_pred)*p_target)

    return pos_err + size_err + confi_err_obj + confi_err_noobj + pred_err

def main():
    args = sys.argv
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

    err_d = loss_d(y, y_pred, D)/tf.cast(D, tf.float32)
    err_w = DECAY*loss_w()
    train = tf.train.GradientDescentOptimizer(1e-4).minimize(err_d + err_w)

    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()

    x_train, y_train = load_train(res_dir, 0, BATCH_SIZE)

    with tf.Session() as sess:
        """
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, model_dir + 'weights.ckpt')
        else:
            sess.run(init)
        """
        sess.run(init)
        print('epoch, training error, test error, weight error')

        for epoch in range(1, 41):
            random.shuffle(parsed_xml_list_train)

            count_train = 0
            while count_train < train_data_size:
                nextcount = min(count_train + BATCH_SIZE, train_data_size)
                x_train, y_train = load_train(res_dir, count_train, nextcount)
                sess.run(train, feed_dict={x: x_train, y: y_train, D: nextcount - count_train, keep_prob: .5})
                count_train = nextcount

            if epoch%5 == 0:
                count_train = 0
                err_train = 0
                while count_train < train_data_size:
                    nextcount = min(count_train + BATCH_SIZE, train_data_size)
                    x_train, y_train = load_train(res_dir, count_train, nextcount)
                    err_train += sess.run(tf.cast(D, tf.float32)*err_d/train_data_size, feed_dict={x: x_train, y: y_train, D: nextcount - count_train, keep_prob: 1.})
                    count_train = nextcount

                count_test = 0
                err_test = 0
                while count_test < test_data_size:
                    nextcount = min(count_test + BATCH_SIZE, test_data_size)
                    x_test, y_test = load_test(res_dir, count_test, nextcount)
                    err_test += sess.run(tf.cast(D, tf.float32)*err_d/test_data_size, feed_dict={x: x_test, y: y_test, D: nextcount - count_test, keep_prob: 1.})
                    count_test = nextcount
                print(epoch, err_train, err_test, sess.run(err_w))
        saver.save(sess, model_dir + 'weights.ckpt')

main()
