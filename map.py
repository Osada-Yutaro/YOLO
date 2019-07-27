import sys
import numpy as np
import tensorflow as tf
import cv2
import yolo
import predictor

def IoU(trgt, pred, sxt, syt, sxp, syp):
    xt, yt, wt, ht, _ = (trgt + [sxt, syt, 0, 0, 0])*[1/7, 1/7, 1, 1, 1]
    xp, yp, wp, hp, _ = (pred + [sxp, syp, 0, 0, 0])*[1/7, 1/7, 1, 1, 1]

    w = min(xt + wt/2, xp + wp/2) - max(xt - wt/2, xp - wp/2)
    h = min(yt + ht/2, yp + hp/2) - max(yt - ht/2, yp - hp/2)
    if w > 0 and h > 0:
        return w*h/(wt*ht + wp*hp - w*h)
    return 0

def precision(trgt, pred, cl, threshold):
    tpfp = 0
    tp = 0
    for sxp in range(7):
        for syp in range(7):
            for bp in range(2):
                if np.argmax(pred[sxp, syp, 0:20]) != cl or pred[sxp, syp, 20 + 5*bp + 4]*pred[sxp, syp, cl] < 0.5:
                    continue
                tpfp += 1
                cont = False
                for sxt in range(7):
                    if cont:
                        continue
                    for syt in range(7):
                        if cont:
                            continue
                        for bt in range(2):
                            if cont:
                                continue
                            if trgt[sxt, syt, cl] != 1. or trgt[sxt, syt, 20 + 5*bt + 4] == 0.:
                                continue
                            iou = IoU(trgt[sxt, syt, 20 + 5*bt:20 + 5*(bt + 1)], pred[sxp, syp, 20 + 5*bp:20 + 5*(bp + 1)], sxt, syt, sxp, syp)
                            if iou >= threshold:
                                tp += 1
                                cont = True
    return tp, tpfp

def find_match(trgt, boundingbox, cl, sxp, syp):
    for sxt in range(7):
        for syt in range(7):
            for bt in range(2):
                if trgt[sxt, syt, cl] == 0. or trgt[sxt, syt, 20 + 5*bt + 4] == 0.:
                    continue
                iou = IoU(trgt[sxt, syt, 20 + 5*bt: 20 + 5*(bt + 1)], boundingbox, sxt, syt, sxp, syp)
                if iou > 0.5:
                    return True
    return False

def count_boxes(trgt, cl):
    res = 0
    for sxt in range(7):
        for syt in range(7):
            if trgt[sxt, syt, cl] == 0.:
                continue
            for bt in range(2):
                if trgt[sxt, syt, 20 + 5*bt + 4] == 1.:
                    res += 1
    return res

def AP(trgt, pred, cl):
    ls = []
    for sxp in range(7):
        for syp in range(7):
            for bp in range(2):
                if np.argmax(pred[sxp, syp, 0:20]) != cl or pred[sxp, syp, 20 + 5*bp + 4]*pred[sxp, syp, cl] < 0.5:
                    continue
                bb = pred[sxp, syp, 20 + 5*bp:20 + 5*(bp + 1)]
                p = pred[sxp, syp, cl]*pred[sxp, syp, 20 + 5*bp + 4]
                ls.append((p, find_match(trgt, bb, cl, sxp, syp)))
    standings = sorted(ls, reverse=True)
    r = 0
    s = 0
    for i in range(len(standings)):
        if standings[i][1]:
            s += (r + 1)/(i + 1)
            r += 1
    if r == 0:
        return -1
    return s/count_boxes(trgt, cl)

def main(saved_model_dir, data_dir):
    with tf.Session() as sess:
        graph_def = tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
        model = graph_def.signature_def['serving_default']
        data = yolo.train.dataset.Data(data_dir)

        for c in range(20):
            print('class:', i)
            batch = 0
            batch_size = 64
            sap = 0
            s = 0
            while batch < data.VALIDATION_DATA_SIZE:
                next_batch = min(batch + batch_size, data.VALIDATION_DATA_SIZE)
                img, trgt = data.load_validation(batch, next_batch)
                pred = sess.run(model.outputs['result'].name, feed_dict={model.inputs['input'].name: img})
                for i in range(next_batch - batch):
                    ap = AP(trgt[i], pred[i], c)
                    if ap != -1:
                        sap += ap
                        s += 1
                batch = next_batch
            print('mAP:', sap/s)
if __name__ == '__main__':
    m = sys.argv[1]
    d = sys.argv[2]
    main(m, d)
