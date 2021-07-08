#!/usr/bin/python3

import tensorflow as tf
import numpy as np

# weights dimensions are W,H, in channels, out channels
weights_data = np.zeros([3,3,3,32], dtype = np.float32)
for i in range(32):
  weights_data[1,1,1,i] = 1.0
weights = tf.constant(weights_data)

weights_output_data = np.zeros([1,1,32,1001], dtype = np.float32)
for i in range(1001):
  weights_output_data[0,0,0,i] = i
  weights_output_data[0,0,1,i] = 10
weights_output = tf.constant(weights_output_data)

input_spec = tf.TensorSpec([1,299,299,3], dtype = np.float32)
strides = [1,2,2,1]
padding="SAME"

def model_forward(x):
  x = tf.nn.conv2d(x, weights, strides = strides, padding = padding)
  x = tf.nn.avg_pool(x, 150, (1,1,1,1), "VALID")
  x = tf.nn.conv2d(x, weights_output, strides = strides, padding = padding)
  x = tf.reshape(x, (1,-1))
  return x

f = tf.function(model_forward)

concrete_func = f.get_concrete_function(input_spec)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

