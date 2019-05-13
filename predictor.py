import glob
import sys
import random

import tensorflow as tf
import numpy as np
from PIL import Image
#import cv2
import yolo

S = yolo.S
B = yolo.B
C = yolo.C
REVERSE_RESOLUTION = yolo.REVERSE_RESOLUTION

threshold = 0.5

def model(x):
    model_dir = sys.argv[2]

    y = yolo.model(x)

    saver = tf.train.Saver()
    sess = tf.InteractiveSession()

    with tf.Session() as sess:
        saver.restore(sess, model_dir + 'weights.ckpt')
        return sess.run(y)

def threshold_processing(image, pred):
    for sx in range(S):
        for sy in range(S):
            max_index = np.ndarray.argmax(pred[0, sx, sy, 0:C])
            pc = pred[0, sx, sy, 0:C][max_index]
            cl = REVERSE_RESOLUTION[max_index]
            for b in range(B):
                x = (sx + pred[0, sx, sy, C + 5*b])/S
                y = (sy + pred[0, sx, sy, C + 5*b + 1])/S
                w = pred[0, sx, sy, C + 5*b + 2]
                h = pred[0, sx, sy, C + 5*b + 3]
                confi = pred[0, sx, sy, C + 5*b + 4]

                if threshold < confi*pc:
                    print('onject:', x, y, w, h, confi, pc, cl)
                else:
                    print('no onject:', confi*pc)


def main():
    image_file = sys.argv[1]
    image = np.array(Image.open(image_file).resize((448, 448)), dtype='float32')
    pred = model(np.reshape(image, (1, 448, 448, 3)))
    threshold_processing(image, pred)

np.set_printoptions(linewidth=np.inf, threshold=np.inf)
main()
