def weight_variable(shape, name):
    import tensorflow as tf
    with tf.variable_scope('yolo', reuse=False):
        w = tf.get_variable(name, initializer=tf.truncated_normal(shape, stddev=0.01))
        return w

def conv2d(x, shape, step=1, name=None):
    import tensorflow as tf
    return tf.nn.conv2d(x, weight_variable(shape, name), [1, step, step, 1], 'SAME')

def leaky_relu(x):
    import tensorflow as tf
    return tf.nn.leaky_relu(x, alpha=0.1)

def max_pool(x, r, step):
    import tensorflow as tf
    return tf.nn.max_pool(x, [1, r, r, 1], [1, step, step, 1], 'SAME')

def batch_normalization(x, shape, name):
    import tensorflow as tf
    with tf.variable_scope('yolo', reuse=False):
        mean = tf.reduce_mean(x, axis=0)
        variance = tf.reduce_mean(tf.square(x - mean))
        offset = tf.get_variable(name + '_offset', initializer=tf.zeros(shape))
        scale = tf.get_variable(name + '_scale', initializer=tf.ones(shape))
        eps = 1e-4
        return scale*(x - mean)/tf.square(variance + eps) + offset

def model(x, keep_prob=1.):
    import tensorflow as tf
    def __first_block(x):
        return max_pool(leaky_relu(conv2d(x, [7, 7, 3, 192], 2, name='w_1')), 2, 2)
    def __second_block(x):
        return max_pool(leaky_relu(conv2d(x, [3, 3, 192, 128], 1, name='w_2')), 2, 2)
    def __third_block(x):
        layer_1 = leaky_relu(conv2d(x, [1, 1, 128, 256], name='w_3_1'))
        layer_2 = leaky_relu(conv2d(layer_1, [3, 3, 256, 256], name='w_3_2'))
        layer_3 = leaky_relu(conv2d(layer_2, [1, 1, 256, 512], name='w_3_3'))
        layer_4 = leaky_relu(conv2d(layer_3, [3, 3, 512, 256], name='w_3_4'))
        return max_pool(layer_4, 2, 2)
    def __fourth_block(x):
        layer_1 = leaky_relu(conv2d(x, [1, 1, 256, 512], name='w_4_1'))
        layer_2 = leaky_relu(conv2d(layer_1, [3, 3, 512, 256], name='w_4_2'))
        layer_3 = leaky_relu(conv2d(layer_2, [1, 1, 256, 512], name='w_4_3'))
        layer_4 = leaky_relu(conv2d(layer_3, [3, 3, 512, 256], name='w_4_4'))
        layer_5 = leaky_relu(conv2d(layer_4, [1, 1, 256, 512], name='w_4_5'))
        layer_6 = leaky_relu(conv2d(layer_5, [3, 3, 512, 256], name='w_4_6'))
        layer_7 = leaky_relu(conv2d(layer_6, [1, 1, 256, 512], name='w_4_7'))
        layer_8 = leaky_relu(conv2d(layer_7, [3, 3, 512, 512], name='w_4_8'))
        layer_9 = leaky_relu(conv2d(layer_8, [1, 1, 512, 1024], name='w_4_9'))
        layer_10 = leaky_relu(conv2d(layer_9, [3, 3, 1024, 512], name='w_4_10'))
        return max_pool(layer_10, 2, 2)
    def __fifth_block(x):
        layer_1 = leaky_relu(conv2d(x, [1, 1, 512, 1024], name='w_5_1'))
        layer_2 = leaky_relu(conv2d(layer_1, [3, 3, 1024, 512], name='w_5_2'))
        layer_3 = leaky_relu(conv2d(layer_2, [1, 1, 512, 1024], name='w_5_3'))
        layer_4 = leaky_relu(conv2d(layer_3, [3, 3, 1024, 1024], name='w_5_4'))
        layer_5 = leaky_relu(conv2d(layer_4, [3, 3, 1024, 1024], name='w_5_5'))
        return leaky_relu(conv2d(layer_5, [3, 3, 1024, 1024], 2, name='w_5_6'))
    def __sixth_block(x):
        layer_1 = leaky_relu(conv2d(x, [3, 3, 1024, 1024], name='w_6_1'))
        return leaky_relu(conv2d(layer_1, [3, 3, 1024, 1024], name='w_6_2'))
    def __seventh_block(x, keep_prob):
        w = weight_variable([7*7*1024, 4096], name='w_7')
        conn = leaky_relu(tf.matmul(tf.reshape(x, [-1, 7*7*1024]), w))
        return tf.nn.dropout(conn, rate=1-keep_prob)
    def __eighth_block(x):
        w = weight_variable([4096, 7*7*30], name='w_8')
        return tf.nn.sigmoid(tf.reshape(tf.matmul(x, w), [-1, 7, 7, 30]))

    return __eighth_block(__seventh_block(__sixth_block(__fifth_block(__fourth_block(__third_block(__second_block(__first_block(x)))))), keep_prob))
