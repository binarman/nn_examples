#!/usr/bin/python3

import tensorflow as tf
import numpy as np
# Construct a basic model.

f = tf.function(lambda x,y: tf.realdiv(x, y))

# Create the concrete function.
input_x = np.ones(shape=[1, 3], dtype=np.complex64)
input_y = np.ones(shape=[3, 1], dtype=np.complex64)
concrete_func = f.get_concrete_function(input_x, input_y)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

