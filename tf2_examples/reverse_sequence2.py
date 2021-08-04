#!/usr/bin/python3

import tensorflow as tf
import numpy as np
# Construct a basic model.

indices = tf.constant([7, 2, 3, 5], dtype=np.int32)
f = tf.function(lambda x: tf.reverse_sequence(x, indices, seq_axis=1))

# Create the concrete function.
input_x = np.ones(shape=[4,8], dtype=np.float32)
concrete_func = f.get_concrete_function(input_x)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

#converter.allow_custom_ops = True

tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

