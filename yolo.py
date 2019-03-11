import xml.etree.ElementTree as ET
import glob

import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split

S = 7
B = 2
C = 20
INDICES = {'diningtable': 0, 'chair': 1, 'horse': 2, 'person': 3, 'tvmonitor': 4, 'bird': 5, 'cow': 6, 'dog': 7, 'bottle': 8, 'pottedplant': 9, 'aeroplane': 10, 'car': 11, 'cat': 12, 'sheep': 13, 'bicycle': 14, 'sofa': 15, 'boat': 16, 'train': 17, 'motorbike': 18, 'bus': 19}
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5
keep_prob = 0.5


def load_dataset():
    input_list = []
    output_list = []

    def vector_in_gridcell(width, height, objs, sx, sy):
        bounding_boxes = np.arange(0, dtype=float)
        confidences = np.zeros(C, dtype=float)
        b = B
        for obj in objs:
            if b == 0:
                continue

            xmin = float(obj.find('bndbox/xmin').text)
            ymin = float(obj.find('bndbox/ymin').text)
            xmax = float(obj.find('bndbox/xmax').text)
            ymax = float(obj.find('bndbox/ymax').text)

            if 0 <= (xmax + xmin)/2 - sx*width/S < width/S and 0 <= (ymax + ymin)/2 - sy*height/S < height/S:
                confidences[INDICES[obj.find('name').text]] = 1.0

                bounding_boxes = np.hstack((
                    bounding_boxes,
                    np.array([S*(xmax + xmin)/2/width - sx,
                              S*(ymax + ymin)/2/height - sy,
                              (xmax - xmin)/width,
                              (ymax - ymin)/height,
                              1.0])))
                b -= 1

        for _ in range(b):
            bounding_boxes = np.hstack((bounding_boxes, np.zeros(5, dtype=float)))
        return np.hstack((bounding_boxes, confidences))

    for parsed_xml in map(ET.parse, glob.glob('VOCdevkit/VOC2007/Annotations/0000*')):
        width = float(parsed_xml.find('size/width').text)
        height = float(parsed_xml.find('size/height').text)

        image = Image.open('VOCdevkit/VOC2007/JPEGImages/' + parsed_xml.find('filename').text).resize((448, 448))
        input_list.append(np.array(image))

        output_tensor = np.zeros((S, S, 5*B + C), dtype=float)
        for sx in range(S):
            for sy in range(S):
                output_tensor[sx][sy] = vector_in_gridcell(width, height, parsed_xml.findall('object'), sx, sy)
        output_list.append(output_tensor)
    return np.array(input_list, dtype='float32'), np.array(output_list, dtype='float32')

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape))

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
    layer_1 = leaky_relu(conv2d(layer_2, [1, 1, 512, 1024]))
    layer_2 = leaky_relu(conv2d(layer_1, [3, 3, 1024, 1024]))
    layer_3 = leaky_relu(conv2d(layer_2, [3, 3, 1024, 1024]))
    return leaky_relu(conv2d(layer_3, [3, 3, 1024, 1024], 2))

def sixth_block(x):
    layer_1 = leaky_relu(conv2d(x, [3, 3, 1024, 1024]))
    return leaky_relu(conv2d(layer_1, [3, 3, 1024, 1024]))

def seventh_block(x):
    w = weight_variable([7*7*1024, 4096])
    conn = leaky_relu(tf.matmul(tf.reshape(x, [-1, 7*7*1024]), w))
    return tf.nn.dropout(conn, keep_prob)

def eighth_block(x):
    w = weight_variable([4096, 7*7*30])
    return tf.reshape(tf.matmul(x, w), [-1, 7, 7, 30])

def loss_5(output_target, output_pred):
    pos_loss = LAMBDA_COORD*output_target[4]*(tf.square(output_target[0] - output_pred[0]) + tf.square(output_target[0] - output_pred[0]))
    size_loss = LAMBDA_COORD*output_target[4]*(tf.square(tf.sqrt(output_target[2]) - output_pred[2]) + tf.square(tf.sqrt(output_target[3]) - output_pred[3]))
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

def loss(output_target, output_pred):
    D = 36
    def upd(x, y):
        return x + 1, y + loss_1(output_target[x], output_pred[x])
    return tf.while_loop(lambda x, y: x < D, upd, (0, 0.))[1]

def main():
    x_data, y_data = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, random_state=1)

    D = x_train.shape[0]

    x = tf.placeholder(tf.float32, [None, 448, 448, 3])
    y = tf.placeholder(tf.float32, [None, S, S, 5*B + C])

    y_pred = eighth_block(seventh_block(sixth_block(fifth_block(fourth_block(third_block(second_block(first_block(x))))))))

    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss(y, y_pred))

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        train.run(feed_dict={x: x_train, y: y_train})
        print(sess.run(y_pred, feed_dict={x: x_train, y: y_train}))
        print(loss(y_train, y_pred).eval(feed_dict={x: x_train}))

main()
