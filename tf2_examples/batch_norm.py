#!/usr/bin/python3

import tensorflow as tf
import numpy as np
# Construct a basic model.

mean = tf.constant([1., 2., 3.])
variance = tf.constant([4., 5., 6.])
offset = tf.constant([10., 11., 12.])
scale = tf.constant([7., 8., 9.])

f = tf.function(lambda x: tf.nn.batch_normalization(x, mean, variance, offset, scale, 1e-5))

# Create the concrete function.
input_data = tf.constant(1., shape=[3, 3])
concrete_func = f.get_concrete_function(input_data)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

