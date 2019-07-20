import sys
import numpy as np
import tensorflow as tf
import cv2

img_file = sys.argv[1]
img = cv2.resize(cv2.imread(img_file), dsize=(448, 448)).astype('float32')
with tf.Session() as sess:
    graph_def = tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING], 'model/SavedModel')
    model = graph_def.signature_def['serving_default']
    pred = sess.run(model.outputs['result'].name, feed_dict={model.inputs['input'].name: np.array([img])})
