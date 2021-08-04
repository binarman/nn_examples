#!/usr/bin/python3

import tensorflow as tf
import numpy as np
# Construct a basic model.

indices = tf.constant([0, 2, 1])

f = tf.function(lambda x: tf.raw_ops.GatherV2(params = x, indices = indices, axis = 1))

# Create the concrete function.
input_data = tf.constant(1., shape=[2, 3, 4])
concrete_func = f.get_concrete_function(input_data)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

