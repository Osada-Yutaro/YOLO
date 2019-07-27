import numpy as np
import cv2
import tensorflow as tf

img = cv2.resize(cv2.imread('VOCdevkit/VOC2007+2012/JPEGImages/2007_000027.jpg'), dsize=(448, 448)).astype('float32')
tfl = tf.lite.Interpreter('model/Lite/tflite_model.tflite')
tfl.allocate_tensors()
input_index = tfl.get_input_details()[0]['index']
output = tfl.tensor(tfl.get_output_details()[0]['index'])
tfl.set_tensor(input_index, np.array([img]))
tfl.invoke()
print(output())
