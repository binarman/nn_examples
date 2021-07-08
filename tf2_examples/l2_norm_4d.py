#!/usr/bin/python3

import tensorflow as tf
import numpy as np

def model_forward(x):
  x = tf.compat.v1.math.l2_normalize(x, dim=-1, epsilon=1e-12)
  return x

f = tf.function(model_forward)

# Create the concrete function.
concrete_func = f.get_concrete_function(tf.TensorSpec([1,8,8,32], dtype=tf.float32))
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

