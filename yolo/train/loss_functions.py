def loss_w():
    import tensorflow as tf
    with tf.variable_scope('yolo', reuse=True):
        err = tf.reduce_sum(tf.square(tf.get_variable('w_1')))
        err += tf.reduce_sum(tf.square(tf.get_variable('w_2')))
        for i in range(1, 5):
            err += tf.reduce_sum(tf.square(tf.get_variable('w_3_' + str(i))))
        for i in range(1, 11):
            err += tf.reduce_sum(tf.square(tf.get_variable('w_4_' + str(i))))
        for i in range(1, 7):
            err += tf.reduce_sum(tf.square(tf.get_variable('w_5_' + str(i))))
        for i in range(1, 3):
            err += tf.reduce_sum(tf.square(tf.get_variable('w_6_' + str(i))))
        err += tf.reduce_sum(tf.square(tf.get_variable('w_7')))
        err += tf.reduce_sum(tf.square(tf.get_variable('w_8')))
        return err

def position_loss(trgt, pred):
    from yolo.architecture import constants as const
    from yolo.train import constants as trcon
    import tensorflow as tf
    B = const.B
    C = const.C
    LAMBDA_COORD = trcon.LAMBDA_COORD
    x_trgt = trgt[:, :, :, C:C + 5*B:5]
    y_trgt = trgt[:, :, :, C + 1:C + 5*B:5]
    confi_trgt = trgt[:, :, :, C + 4:C + 5*B:5]

    x_pred = pred[:, :, :, C:C + 5*B:5]
    y_pred = pred[:, :, :, C + 1:C + 5*B:5]

    x_loss = LAMBDA_COORD*tf.reduce_sum(tf.square(x_trgt - x_pred)*confi_trgt)
    y_loss = LAMBDA_COORD*tf.reduce_sum(tf.square(y_trgt - y_pred)*confi_trgt)
    return x_loss + y_loss

def size_loss(trgt, pred):
    from yolo.architecture import constants as const
    from yolo.train import constants as trcon
    import tensorflow as tf
    B = const.B
    C = const.C
    LAMBDA_COORD = trcon.LAMBDA_COORD
    w_trgt = trgt[:, :, :, C + 2:C + 5*B:5]
    h_trgt = trgt[:, :, :, C + 3:C + 5*B:5]
    confi_trgt = trgt[:, :, :, C + 4:C + 5*B:5]

    w_pred = tf.nn.relu(pred[:, :, :, C + 2:C + 5*B:5])
    h_pred = tf.nn.relu(pred[:, :, :, C + 3:C + 5*B:5])

    eps = 1e-1
    w_loss = LAMBDA_COORD*tf.reduce_sum(
        tf.square(tf.sqrt(w_trgt + eps) - tf.sqrt(w_pred + eps))*confi_trgt
        )
    h_loss = LAMBDA_COORD*tf.reduce_sum(
        tf.square(tf.sqrt(h_trgt + eps) - tf.sqrt(h_pred + eps))*confi_trgt
        )
    return w_loss + h_loss

def confidence_loss(trgt, pred):
    from yolo.architecture import constants as const
    from yolo.train import constants as trcon
    import tensorflow as tf
    B = const.B
    C = const.C
    LAMBDA_NOOBJ = trcon.LAMBDA_NOOBJ
    confi_trgt = trgt[:, :, :, C + 4:C + 5*B:5]

    confi_pred = pred[:, :, :, C + 4:C + 5*B:5]

    confi_loss_obj = tf.reduce_sum(tf.square(confi_trgt - confi_pred)*confi_trgt)
    confi_loss_noobj = LAMBDA_NOOBJ*tf.reduce_sum(
        tf.square(confi_trgt - confi_pred)*(1 - confi_trgt)
        )

    return confi_loss_obj + confi_loss_noobj

def class_loss(trgt, pred, D):
    from yolo.architecture import constants as const
    import tensorflow as tf
    S = const.S
    C = const.C
    p_trgt = trgt[:, :, :, 0:C]
    p_pred = pred[:, :, :, 0:C]
    return tf.reduce_sum(
        tf.square(p_trgt - p_pred)*tf.reshape(tf.reduce_max(p_trgt, axis=[3]), [D, S, S, 1])
        )

def loss_d(trgt, pred, D):
    return (position_loss(trgt, pred) +
            size_loss(trgt, pred) +
            confidence_loss(trgt, pred) +
            class_loss(trgt, pred, D))
