def __first_block(x):
    from yolo.architecture import layer_functions as lf
    return lf.max_pool(lf.leaky_relu(lf.conv2d(x, [7, 7, 3, 192], 2, name='w_1')), 2, 2)

def __second_block(x):
    from yolo.architecture import layer_functions as lf
    layer_0 = __first_block(x)
    return lf.max_pool(lf.leaky_relu(lf.conv2d(layer_0, [3, 3, 192, 128], 1, name='w_2')), 2, 2)

def __third_block(x):
    from yolo.architecture import layer_functions as lf
    layer_0 = __second_block(x)
    layer_1 = lf.leaky_relu(lf.conv2d(layer_0, [1, 1, 128, 256], name='w_3_1'))
    layer_2 = lf.leaky_relu(lf.conv2d(layer_1, [3, 3, 256, 256], name='w_3_2'))
    layer_3 = lf.leaky_relu(lf.conv2d(layer_2, [1, 1, 256, 512], name='w_3_3'))
    layer_4 = lf.leaky_relu(lf.conv2d(layer_3, [3, 3, 512, 256], name='w_3_4'))
    return lf.max_pool(layer_4, 2, 2)

def __fourth_block(x):
    from yolo.architecture import layer_functions as lf
    layer_0 = __third_block(x)
    layer_1 = lf.leaky_relu(lf.conv2d(layer_0, [1, 1, 256, 512], name='w_4_1'))
    layer_2 = lf.leaky_relu(lf.conv2d(layer_1, [3, 3, 512, 256], name='w_4_2'))
    layer_3 = lf.leaky_relu(lf.conv2d(layer_2, [1, 1, 256, 512], name='w_4_3'))
    layer_4 = lf.leaky_relu(lf.conv2d(layer_3, [3, 3, 512, 256], name='w_4_4'))
    layer_5 = lf.leaky_relu(lf.conv2d(layer_4, [1, 1, 256, 512], name='w_4_5'))
    layer_6 = lf.leaky_relu(lf.conv2d(layer_5, [3, 3, 512, 256], name='w_4_6'))
    layer_7 = lf.leaky_relu(lf.conv2d(layer_6, [1, 1, 256, 512], name='w_4_7'))
    layer_8 = lf.leaky_relu(lf.conv2d(layer_7, [3, 3, 512, 512], name='w_4_8'))
    layer_9 = lf.leaky_relu(lf.conv2d(layer_8, [1, 1, 512, 1024], name='w_4_9'))
    layer_10 = lf.leaky_relu(lf.conv2d(layer_9, [3, 3, 1024, 512], name='w_4_10'))
    return lf.max_pool(layer_10, 2, 2)

def __fifth_block(x):
    from yolo.architecture import layer_functions as lf
    layer_0 = __fourth_block(x)
    layer_1 = lf.leaky_relu(lf.conv2d(layer_0, [1, 1, 512, 1024], name='w_5_1'))
    layer_2 = lf.leaky_relu(lf.conv2d(layer_1, [3, 3, 1024, 512], name='w_5_2'))
    layer_3 = lf.leaky_relu(lf.conv2d(layer_2, [1, 1, 512, 1024], name='w_5_3'))
    layer_4 = lf.leaky_relu(lf.conv2d(layer_3, [3, 3, 1024, 1024], name='w_5_4'))
    layer_5 = lf.leaky_relu(lf.conv2d(layer_4, [3, 3, 1024, 1024], name='w_5_5'))
    return lf.leaky_relu(lf.conv2d(layer_5, [3, 3, 1024, 1024], 2, name='w_5_6'))

def __sixth_block(x):
    from yolo.architecture import layer_functions as lf
    layer_0 = __fifth_block(x)
    layer_1 = lf.leaky_relu(lf.conv2d(layer_0, [3, 3, 1024, 1024], name='w_6_1'))
    return lf.leaky_relu(lf.conv2d(layer_1, [3, 3, 1024, 1024], name='w_6_2'))

def __seventh_block(x, keep_prob):
    from yolo.architecture import layer_functions as lf
    import tensorflow as tf
    layer_0 = __sixth_block(x)
    w = lf.weight_variable([7*7*1024, 4096], name='w_7')
    conn = lf.leaky_relu(tf.matmul(tf.reshape(layer_0, [-1, 7*7*1024]), w))
    return tf.nn.dropout(conn, rate=1-keep_prob)

def __eighth_block(x, keep_prob):
    from yolo.architecture import layer_functions as lf
    import tensorflow as tf
    layer_0 = __seventh_block(x, keep_prob)
    w = lf.weight_variable([4096, 7*7*30], name='w_8')
    return tf.nn.relu(tf.reshape(tf.matmul(layer_0, w), [-1, 7, 7, 30]), 'result')

def graph_def(x, keep_prob=1.):
    return __eighth_block(x, keep_prob)
