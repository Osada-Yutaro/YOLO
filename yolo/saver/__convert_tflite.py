def convert_tflite(saved_model_dir, tflite_path, quantize=True):
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(
        saved_model_dir,
        signature_key='serving_default')
    if quantize:
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    tflite_model = converter.convert()
    open(tflite_path, 'wb').write(tflite_model)
