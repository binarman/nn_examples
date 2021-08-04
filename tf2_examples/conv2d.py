#!/usr/bin/python3

import tensorflow as tf
import numpy as np

input_size = tf.constant([1,4,4,2])

# weights dimensions are W,H, in channels, out channels
weights = tf.constant(np.ones([3,3,3,1], dtype = np.float32))
input_spec = tf.TensorSpec([1,16,16,3], dtype = np.float32)
strides = [1,1,1,1]
padding="SAME"

f = tf.function(lambda x: tf.nn.conv2d(x, weights, strides = strides, padding = padding))

concrete_func = f.get_concrete_function(input_spec)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

