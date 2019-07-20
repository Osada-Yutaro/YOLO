def save_model(checkpoint_dir, saved_model_dir):
    from .architecture.network import graph_def
    import os
    import tensorflow as tf
    def _build_signature(sig_inputs, sig_outputs):
        return tf.saved_model.signature_def_utils.build_signature_def(sig_inputs, sig_outputs, tf.saved_model.signature_constants.REGRESS_METHOD_NAME)
    x = tf.placeholder(tf.float32, [None, 448, 448, 3], 'input')
    y = graph_def(x, 1.)
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(checkpoint_dir, 'weights.ckpt'))
        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
        def_map = {'yolo': _build_signature({'input': tf.saved_model.utils.build_tensor_info(x)}, {'result': tf.saved_model.utils.build_tensor_info(y)})}
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map=def_map)
        builder.save()

def convert_tflite(saved_model_dir, tflite_dir):
    import os
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir, signature_key='yolo')
    tflite_model = converter.convert()
    open(os.path.join(tflite_dir, 'tflite_model.tflite'), 'wb').write(tflite_model)
