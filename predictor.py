import os
import glob
import sys
import random

import tensorflow as tf
import numpy as np
import cv2
import yolo

S = yolo.architecture.constants.S
B = yolo.architecture.constants.B
C = yolo.architecture.constants.C
REVERSE_RESOLUTION = yolo.architecture.constants.REVERSE_RESOLUTION

threshold = 0.5

def bounding_box(img, bb, cl, color):
    img_cp = np.copy(img)

    x, y, w, h = bb

    height = img_cp.shape[0]
    width = img_cp.shape[1]

    xmin = max(int((x - w*0.5)*width), 0)
    xmax = min(int((x + w*0.5)*width), width - 1)
    ymin = max(int((y - h*0.5)*height), 0)
    ymax = min(int((y + h*0.5)*height), height - 1)
    cv2.line(img_cp, (xmin, ymin), (xmin, ymax), color, 1)
    cv2.line(img_cp, (xmin, ymax), (xmax, ymax), color, 1)
    cv2.line(img_cp, (xmax, ymax), (xmax, ymin), color, 1)
    cv2.line(img_cp, (xmax, ymin), (xmin, ymin), color, 1)
    cv2.putText(img_cp, cl, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return img_cp

def threshold_processing(image, pred):
    img_cp = np.copy(image)
    for sx in range(S):
        for sy in range(S):
            max_index = np.ndarray.argmax(pred[sx, sy, 0:C])
            pc = pred[sx, sy, 0:C][max_index]
            cl = REVERSE_RESOLUTION[max_index]
            for b in range(B):
                x = (sx + pred[sx, sy, C + 5*b])/S
                y = (sy + pred[sx, sy, C + 5*b + 1])/S
                w = pred[sx, sy, C + 5*b + 2]
                h = pred[sx, sy, C + 5*b + 3]
                confi = pred[sx, sy, C + 5*b + 4]

                if threshold < confi*pc:
                    img_cp = bounding_box(img_cp, (x, y, w, h), cl, (0, 255, 0))
                    #print('onject:', x, y, w, h, confi, pc, cl, confi*pc)
    return img_cp

def main():
    with tf.Session() as sess:
        graph_def = tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING], 'model/SavedModel')
        model = graph_def.signature_def['serving_default']
        for i in range(16):
            image, _ = yolo.train.dataset.Data('VOCdevkit/VOC2007+2012/').load_validation(i*16, (i + 1)*16)
            pred = sess.run(model.outputs['result'].name, feed_dict={model.inputs['input'].name: image})
            for j in range(16):
                res = threshold_processing(image[j], pred[j])
                cv2.imwrite('valid_images/predict' + str(i) + '_' + str(j) + '.png', res)

if __name__ == 'main':
    main()
