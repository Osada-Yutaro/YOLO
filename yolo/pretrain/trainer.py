def fit(data_dir, checkpoint_dir, epoch_size=10, lr=1e-4, start_epoch=1):
    from .architecture import network
    from . import dataset
    import os
    import random
    import tensorflow as tf

    BATCH_SIZE = 16

    x = tf.compat.v1.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.compat.v1.placeholder(tf.float32, [None, 1000])
    learning_rate = tf.compat.v1.placeholder(tf.float32)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    y_pred = network.graph_def(x)

    err_d = tf.reduce_sum(tf.square(y - y_pred))/BATCH_SIZE
    minimize = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(err_d)

    saver = tf.compat.v1.train.Saver()

    init = tf.compat.v1.global_variables_initializer()

    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    data = dataset.Data(data_dir)

    with tf.compat.v1.Session(config=config) as sess:
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
                x_train, y_train = data.load_train(count_train, nextcount)
                sess.run(
                    minimize,
                    feed_dict={
                        x: x_train,
                        y: y_train,
                        learning_rate: lr})
                count_train = nextcount

            if epoch%10 == 0:
                count_train = 0
                err_train = 0
                while count_train < data.TRAIN_DATA_SIZE:
                    nextcount = min(count_train + BATCH_SIZE, data.TRAIN_DATA_SIZE)
                    x_train, y_train = data.load_train(count_train, nextcount)
                    err_train += sess.run(
                        err_d*BATCH_SIZE/data.TRAIN_DATA_SIZE,
                        feed_dict={
                            x: x_train,
                            y: y_train})
                    count_train = nextcount

                count_validation = 0
                err_validation = 0
                while count_validation < data.VALIDATION_DATA_SIZE:
                    nextcount = min(count_validation + BATCH_SIZE, data.VALIDATION_DATA_SIZE)
                    x_validation, y_validation = data.load_validation(count_validation, nextcount)
                    err_validation += sess.run(
                        err_d*BATCH_SIZE/data.VALIDATION_DATA_SIZE,
                        feed_dict={
                            x: x_validation,
                            y: y_validation})
                    count_validation = nextcount

                print(epoch, err_train, err_validation)
        saver.save(sess, os.path.join(checkpoint_dir, 'weights.ckpt'))
