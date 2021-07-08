#!/usr/bin/python3

import tensorflow as tf
import numpy as np
# Construct a basic model.
f = tf.function(lambda x: tf.reshape(x, (-1,)))

# Create the concrete function.
concrete_func = f.get_concrete_function(tf.TensorSpec((1,4,4,1), dtype=tf.float32))
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

converter.allow_custom_ops = True

tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

