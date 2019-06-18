import numpy as np
import tensorflow as tf
import yolo

res_dir = '../kw_resources/VOCdevkit/VOC2007+2012/'
model_dir = '../kw_resources/model/'
classify_count = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float')
ideal_count = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float')
confi_count = 0.
ideal_confi_count = 0.
yolo.load_dataset(res_dir)

x = tf.placeholder(tf.float32, [None, 448, 448, 3])
pred = yolo.model(x)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, model_dir + 'weights.ckpt')
    res = sess.run(pred)
    for i in range(0, yolo.VALIDATION_DATA_SIZE, 16):
        _x, _y = yolo.load_validation(res_dir, i, min(i + 16, yolo.VALIDATION_DATA_SIZE))
        res = sess.run(pred, feed_dict={x: _x})
        for d in range(16):
            for sx in range(7):
                for sy in range(7):
                    if _y[d, sx, sy, 24] == 1.:
                        ideal_confi_count += 1
                        confi_count += res[d, sx, sy, 24]
                    for c in range(20):
                        if _y[d, sx, sy, c] == 1.:
                            classify_count[c] += res[d, sx, sy, c]
                            ideal_count[c] += 1

print(classify_count/ideal_count)
print(confi_count/ideal_confi_count)
