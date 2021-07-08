#!/usr/bin/python3

import tensorflow as tf
import numpy as np

# weights dimensions are W,H, in channels, out channels
weights_data = np.ones([3,3,3,32], dtype = np.float32)
weights = tf.constant(weights_data)

input_spec = tf.TensorSpec([1,16,16,3], dtype = np.float32)
strides = [1,2,2,1]
padding="SAME"

def model_forward(x):
  x = tf.nn.conv2d(x, weights, strides = strides, padding = padding)
  x = tf.math.reduce_mean(x, axis=[1,2])
  x = tf.reshape(x, (1,-1))
  return x

f = tf.function(model_forward)

concrete_func = f.get_concrete_function(input_spec)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

