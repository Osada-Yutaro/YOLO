def weight_variable(shape, name):
    import tensorflow as tf
    with tf.compat.v1.variable_scope('yolo', reuse=tf.compat.v1.AUTO_REUSE):
        w = tf.compat.v1.get_variable(
            name, initializer=tf.random.truncated_normal(shape, stddev=0.03)
            )
        return w

def conv2d(x, shape, step=1, name=None):
    import tensorflow as tf
    return tf.nn.conv2d(x, weight_variable(shape, name), [1, step, step, 1], 'SAME')

def leaky_relu(x):
    import tensorflow as tf
    return tf.nn.leaky_relu(x, alpha=0.1)

def max_pool(x, r, step):
    import tensorflow as tf
    return tf.nn.max_pool2d(x, [1, r, r, 1], [1, step, step, 1], 'SAME')

def avg_pool(x, r, step):
    import tensorflow as tf
    return tf.nn.avg_pool2d(x, [1, r, r, 1], [1, step, step, 1], 'SAME')

def batch_normalization(x, shape, name):
    import tensorflow as tf
    with tf.variable_scope('yolo', reuse=tf.AUTO_REUSE):
        mean = tf.reduce_mean(x, axis=0)
        variance = tf.reduce_mean(tf.square(x - mean))
        offset = tf.get_variable(name + '_offset', initializer=tf.zeros(shape))
        scale = tf.get_variable(name + '_scale', initializer=tf.ones(shape))
        eps = 1e-4
        return tf.nn.batch_normalization(x, mean, variance, offset, scale, eps)

def batch_normalized_conv2d(x, shape, step=1, name=None):
    conv = conv2d(x, shape, step, name)
    return batch_normalization(conv2d(x, shape, step, name), conv.shape[1:], name)
