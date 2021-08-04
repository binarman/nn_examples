#!/usr/bin/python3

import tensorflow as tf
import numpy as np
# Construct a basic model.

f = tf.function(lambda x,y: tf.linalg.matmul(x, y, transpose_a=True, transpose_b=False))
#f = tf.function(lambda x,y: tf.matmul(x, y))

#transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False,
#    a_is_sparse=False, b_is_sparse=False, name=None

# Create the concrete function.
input_x = np.ones(shape=[3, 3], dtype=np.complex64)
input_y = np.ones(shape=[3, 3], dtype=np.complex64)
concrete_func = f.get_concrete_function(input_x, input_y)

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.allow_custom_ops = True

tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

