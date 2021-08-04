#!/usr/bin/python3

import tensorflow as tf
import numpy as np

input_size = tf.constant([1,4,4,2])

# weights dimensions are W,H, in channels, out channels
weights = tf.constant(np.ones([3,3,2,3], dtype = np.float32))
input_data = np.ones([1,2,2,3], dtype = np.float32)
strides = [1,1,1,1]
padding="VALID"

f = tf.function(lambda x: tf.raw_ops.Conv2DBackpropInput(input_sizes = input_size,
                                                              filter = weights,
                                                              out_backprop = x,
                                                              strides = strides,
                                                              padding = padding))

concrete_func = f.get_concrete_function(input_data)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

