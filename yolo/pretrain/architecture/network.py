def graph_def(x):
    import tensorflow as tf
    from yolo.architecture import layer_functions as lf
    from yolo.architecture.network import __fourth_block
    def __fifth_block(x):
        layer_0 = __fourth_block(x)
        layer_1 = lf.leaky_relu(lf.conv2d(layer_0, [1, 1, 1024, 512], name='w_5_1'))
        layer_2 = lf.leaky_relu(lf.conv2d(layer_1, [3, 3, 512, 1024], name='w_5_2'))
        layer_3 = lf.leaky_relu(lf.conv2d(layer_2, [1, 1, 1024, 512], name='w_5_3'))
        layer_4 = lf.leaky_relu(lf.conv2d(layer_3, [3, 3, 512, 1024], name='w_5_4'))
        layer_5 = tf.reduce_mean(layer_4, [1, 2])
        w = lf.weight_variable([1024, 1000], name='pre_w_6')
        return lf.leaky_relu(tf.matmul(tf.reshape(layer_5, [-1, 1024]), w))
    return __fifth_block(x)
