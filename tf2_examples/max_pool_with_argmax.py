#!/usr/bin/python3

import tensorflow as tf
import numpy as np
# Construct a basic model.

f = tf.function(lambda x: tf.nn.max_pool_with_argmax(x, ksize=3, strides=1, padding="SAME", output_dtype=tf.int32))

# Create the concrete function.
input_data = tf.constant(1., shape=[1, 4, 4, 32])
concrete_func = f.get_concrete_function(input_data)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

converter.allow_custom_ops = True

tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

