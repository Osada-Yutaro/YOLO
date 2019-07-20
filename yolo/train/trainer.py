def fit(data_dir, checkpoint_dir, epoch_size=10, lr=1e-4, start_epoch=1):
    from ..architecture import constants as acons
    from ..architecture import network
    from . import constants as tcons
    from . import dataset
    from . import loss_functions
    import os
    import random
    import tensorflow as tf

    S = acons.S
    B = acons.B
    C = acons.C
    BATCH_SIZE = tcons.BATCH_SIZE

    x = tf.placeholder(tf.float32, [None, 448, 448, 3])
    y = tf.placeholder(tf.float32, [None, S, S, 5*B + C])
    D = tf.placeholder(tf.int32)
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    y_pred = network.graph_def(x, keep_prob)

    err_d = loss_functions.loss_d(y, y_pred, D)/BATCH_SIZE
    minimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(err_d)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    data = dataset.Data(data_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, os.path.join(checkpoint_dir, 'weights.ckpt'))
        else:
            sess.run(init)
        random.seed()
        if start_epoch == 1:
            print('#epoch, training error, validation error')
        for epoch in range(start_epoch, start_epoch + epoch_size):
            data.shuffle()

            count_train = 0
            while count_train < data.TRAIN_DATA_SIZE:
                nextcount = min(count_train + BATCH_SIZE, data.TRAIN_DATA_SIZE)
                x_train, y_train = data.load_train(count_train, nextcount, prepro=True)
                sess.run(minimize, feed_dict={x: x_train, y: y_train, D: nextcount - count_train, keep_prob: .5, learning_rate: lr})
                count_train = nextcount

            if epoch%10 == 0:
                count_train = 0
                err_train = 0
                while count_train < data.TRAIN_DATA_SIZE:
                    nextcount = min(count_train + BATCH_SIZE, data.TRAIN_DATA_SIZE)
                    x_train, y_train = data.load_train(count_train, nextcount, prepro=False)
                    err_train += sess.run(err_d*BATCH_SIZE/data.TRAIN_DATA_SIZE, feed_dict={x: x_train, y: y_train, D: nextcount - count_train, keep_prob: 1.})
                    count_train = nextcount

                count_validation = 0
                err_validation = 0
                while count_validation < data.VALIDATION_DATA_SIZE:
                    nextcount = min(count_validation + BATCH_SIZE, data.VALIDATION_DATA_SIZE)
                    x_validation, y_validation = data.load_validation(count_validation, nextcount)
                    err_validation += sess.run(err_d*BATCH_SIZE/data.VALIDATION_DATA_SIZE, feed_dict={x: x_validation, y: y_validation, D: nextcount - count_validation, keep_prob: 1.})
                    count_validation = nextcount

                print(epoch, err_train, err_validation)
        saver.save(sess, os.path.join(checkpoint_dir, 'weights.ckpt'))
