#!/usr/bin/python3

import tensorflow as tf
import numpy as np
# Construct a basic model.
root = tf.train.Checkpoint()

root.f = tf.function(lambda x, axis: tf.reverse(x, axis))

# Create the concrete function.
input_x = np.ones(shape=[3,4], dtype=np.float32)
axis = np.array([0,1], dtype=np.int32)
concrete_func = root.f.get_concrete_function(input_x, axis)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

#converter.allow_custom_ops = True

tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

