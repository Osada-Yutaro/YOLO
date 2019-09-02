def graph_def(x):
    import tensorflow as tf
    from yolo.architecture import layer_functions as lf
    from yolo.architecture.network import __fifth_block
    def __sixth_block(x):
        layer_0 = __fifth_block(x)
        layer_1 = lf.avg_pool(layer_0, 1, 1)
        w = lf.weight_variable([7*7*1024, 1000], name='pre_w_6')
        return lf.leaky_relu(tf.matmul(tf.reshape(layer_1, [-1, 7*7*1024]), w))
    return __sixth_block(x)
