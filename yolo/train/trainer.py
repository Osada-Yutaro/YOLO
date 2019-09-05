def restore(sess, saver, init, checkpoint_dir):
    import os
    import tensorflow as tf
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, os.path.join(checkpoint_dir, 'weights.ckpt'))
    else:
        sess.run(init)


def fit(data_dir, checkpoint_dir, epoch_size=10, lr=1e-4, start_epoch=1):
    from yolo.architecture import constants as acons
    from yolo.architecture import network
    from yolo.train import constants as tcons
    from yolo.train import loss_functions
    from yolo import dataset
    import os
    import random
    import tensorflow as tf

    tf.compat.v1.disable_v2_behavior()

    x = tf.compat.v1.placeholder(tf.float32, [None, 448, 448, 3])
    y = tf.compat.v1.placeholder(tf.float32, [None, acons.S, acons.S, 5*acons.B + acons.C])
    D = tf.compat.v1.placeholder(tf.int32)
    keep_prob = tf.compat.v1.placeholder(tf.float32)
    learning_rate = tf.compat.v1.placeholder(tf.float32)

    y_pred = network.graph_def(x, keep_prob)

    err_d = loss_functions.loss_d(y, y_pred, D)/tcons.BATCH_SIZE
    minimize = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(err_d)

    saver = tf.compat.v1.train.Saver()

    init = tf.compat.v1.global_variables_initializer()

    data = dataset.VOCData(data_dir)

    with tf.compat.v1.Session() as sess:
        restore(sess, saver, init, checkpoint_dir)
        random.seed()
        if start_epoch == 1:
            print('#epoch, training error, validation error')
        for epoch in range(start_epoch, start_epoch + epoch_size):
            data.shuffle()

            count_train = 0
            while count_train < data.TRAIN_DATA_SIZE:
                nextcount = min(count_train + tcons.BATCH_SIZE, data.TRAIN_DATA_SIZE)
                x_train, y_train = data.load_train(count_train, nextcount, prepro=True)
                sess.run(minimize, feed_dict={
                    x: x_train,
                    y : y_train,
                    D: nextcount - count_train,
                    keep_prob: .5,
                    learning_rate: lr
                    })
                count_train = nextcount

            if epoch%10 == 0:
                count_train = 0
                err_train = 0
                while count_train < data.TRAIN_DATA_SIZE:
                    nextcount = min(count_train + tcons.BATCH_SIZE, data.TRAIN_DATA_SIZE)
                    x_train, y_train = data.load_train(count_train, nextcount, prepro=False)
                    err_train += sess.run(
                        err_d*tcons.BATCH_SIZE/data.TRAIN_DATA_SIZE,
                        feed_dict={
                            x: x_train,
                            y: y_train,
                            D: nextcount - count_train,
                            keep_prob: 1.})
                    count_train = nextcount

                count_validation = 0
                err_validation = 0
                while count_validation < data.VALIDATION_DATA_SIZE:
                    nextcount = min(count_validation + tcons.BATCH_SIZE, data.VALIDATION_DATA_SIZE)
                    x_validation, y_validation = data.load_validation(count_validation, nextcount)
                    err_validation += sess.run(
                        err_d*tcons.BATCH_SIZE/data.VALIDATION_DATA_SIZE,
                        feed_dict={
                            x: x_validation,
                            y: y_validation,
                            D: nextcount - count_validation,
                            keep_prob: 1.})
                    count_validation = nextcount

                print(epoch, err_train, err_validation)
        saver.save(sess, os.path.join(checkpoint_dir, 'weights.ckpt'))
