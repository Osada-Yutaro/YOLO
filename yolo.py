import xml.etree.ElementTree as ET
import glob
import gc

import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter

S = 7
B = 2
C = 20
INDICES = {'diningtable': 0, 'chair': 1, 'horse': 2, 'person': 3, 'tvmonitor': 4, 'bird': 5, 'cow': 6, 'dog': 7, 'bottle': 8, 'pottedplant': 9, 'aeroplane': 10, 'car': 11, 'cat': 12, 'sheep': 13, 'bicycle': 14, 'sofa': 15, 'boat': 16, 'train': 17, 'motorbike': 18, 'bus': 19}
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5
keep_prob = 0.5
BATCH_SIZE = 64


def load_dataset(train_size=0.75):
    D = len(glob.glob('VOCdevkit/VOC2007/Annotations/*'))
    x_data = np.zeros((D, 448, 448, 3), dtype='float32')
    y_data = np.zeros((D, S, S, 5*B + C), dtype='float32')

    d = 0
    for parsed_xml in map(ET.parse, glob.glob('VOCdevkit/VOC2007/Annotations/*')):
        width = float(parsed_xml.find('size/width').text)
        height = float(parsed_xml.find('size/height').text)

        x_data[d] = np.array(Image.open('VOCdevkit/VOC2007/JPEGImages/' + parsed_xml.find('filename').text).resize((448, 448)))

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
                        y_data[d][sx][sy][5*b] = S*(xmax + xmin)/2/width - sx
                        y_data[d][sx][sy][5*b + 1] = S*(ymax + ymin)/2/height - sy
                        y_data[d][sx][sy][5*b + 2] = (xmax - xmin)/width
                        y_data[d][sx][sy][5*b + 3] = (ymax - ymin)/height
                        y_data[d][sx][sy][5*b + 4] = 1.

                        b += 1
        d += 1

    return x_data[0:int(D*train_size)], x_data[int(D*train_size):D], y_data[0:int(D*train_size)], y_data[int(D*train_size):D]

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.03))

def conv2d(x, shape, step=1):
    return tf.nn.conv2d(x, weight_variable(shape), [1, step, step, 1], 'SAME')

def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.1)

def max_pool(x, r, step):
    return tf.nn.max_pool(x, [1, r, r, 1], [1, step, step, 1], 'SAME')

def first_block(x):
    return max_pool(leaky_relu(conv2d(x, [7, 7, 3, 192], 2)), 2, 2)

def second_block(x):
    return max_pool(leaky_relu(conv2d(x, [3, 3, 192, 128], 1)), 2, 2)

def third_block(x):
    layer_1 = leaky_relu(conv2d(x, [1, 1, 128, 256]))
    layer_2 = leaky_relu(conv2d(layer_1, [3, 3, 256, 256]))
    layer_3 = leaky_relu(conv2d(layer_2, [1, 1, 256, 512]))
    layer_4 = leaky_relu(conv2d(layer_3, [3, 3, 512, 256]))
    return max_pool(layer_4, 2, 2)

def fourth_block(x):
    layer_1 = leaky_relu(conv2d(x, [1, 1, 256, 512]))
    layer_2 = leaky_relu(conv2d(layer_1, [3, 3, 512, 256]))
    for _ in range(2):
        layer_1 = leaky_relu(conv2d(layer_2, [1, 1, 256, 512]))
        layer_2 = leaky_relu(conv2d(layer_1, [3, 3, 512, 256]))
    layer_1 = leaky_relu(conv2d(layer_2, [1, 1, 256, 512]))
    layer_2 = leaky_relu(conv2d(layer_1, [3, 3, 512, 512]))

    layer_3 = leaky_relu(conv2d(layer_2, [1, 1, 512, 1024]))
    layer_4 = leaky_relu(conv2d(layer_3, [3, 3, 1024, 512]))
    return max_pool(layer_4, 2, 2)

def fifth_block(x):
    layer_1 = leaky_relu(conv2d(x, [1, 1, 512, 1024]))
    layer_2 = leaky_relu(conv2d(layer_1, [3, 3, 1024, 512]))
    layer_3 = leaky_relu(conv2d(layer_2, [1, 1, 512, 1024]))
    layer_4 = leaky_relu(conv2d(layer_3, [3, 3, 1024, 1024]))
    layer_5 = leaky_relu(conv2d(layer_4, [3, 3, 1024, 1024]))
    return leaky_relu(conv2d(layer_5, [3, 3, 1024, 1024], 2))

def sixth_block(x):
    layer_1 = leaky_relu(conv2d(x, [3, 3, 1024, 1024]))
    return leaky_relu(conv2d(layer_1, [3, 3, 1024, 1024]))

def seventh_block(x):
    w = weight_variable([7*7*1024, 4096])
    conn = leaky_relu(tf.matmul(tf.reshape(x, [-1, 7*7*1024]), w))
    return tf.nn.dropout(conn, keep_prob)

def eighth_block(x):
    w = weight_variable([4096, 7*7*30])
    return tf.nn.relu(tf.reshape(tf.matmul(x, w), [-1, 7, 7, 30]))

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

def loss(output_target, output_pred, D):
    def upd(x, y):
        return x + 1, y + loss_1(output_target[x], output_pred[x])
    return tf.while_loop(lambda x, y: x < D, upd, (0, 0.))[1]

def main():
    x_train, x_test, y_train, y_test = load_dataset()

    x = tf.placeholder(tf.float32, [None, 448, 448, 3])
    y = tf.placeholder(tf.float32, [None, S, S, 5*B + C])
    D = tf.placeholder(tf.int32)

    train_data_size = x_train.shape[0]
    test_data_size = x_test.shape[0]

    y_pred = eighth_block(seventh_block(sixth_block(fifth_block(fourth_block(third_block(second_block(first_block(x))))))))

    err = loss(y, y_pred, D)
    train = tf.train.GradientDescentOptimizer(1e-7).minimize(err)

    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(135):
            count_train = 0
            err_train = 0
            while count_train < train_data_size:
                nextcount = min(count_train + BATCH_SIZE, train_data_size)
                sess.run(train, feed_dict={x: x_train[count_train:nextcount], y: y_train[count_train:nextcount], D: nextcount - count_train})
                err_train += sess.run(err, feed_dict={x: x_train[count_train:nextcount], y: y_train[count_train:nextcount], D: nextcount - count_train})
                count_train = nextcount

            count_test = 0
            err_test = 0
            while count_test < test_data_size:
                nextcount = min(count_test + BATCH_SIZE, test_data_size)
                err_test += sess.run(err, feed_dict={x: x_test[count_test:nextcount], y: y_test[count_test:nextcount], D: nextcount - count_test})
                count_test = nextcount
            print('epoch=', epoch, ', ', 'train error=', err_train, ', ', 'validation error=', err_test, sep='')
        saver.save(sess, 'model.ckpt')

main()
