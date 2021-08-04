#!/usr/bin/python3

import tensorflow as tf
import numpy as np

# weights dimensions are W,H, in channels, out channels
input_spec = tf.TensorSpec([1, 128, 128, 3], dtype = np.float32)

def separable_conv(x, input_channels, output_channels, stride):
  depthwise_weights = np.random.random((3, 3, input_channels, 1)).astype(np.float32)
  pointwise_weights = np.random.random((1, 1, input_channels, output_channels)).astype(np.float32)
  x = tf.nn.depthwise_conv2d(x, depthwise_weights, strides = (1,stride,stride,1), padding = "SAME")
  x = tf.nn.relu6(x)
  x = tf.nn.conv2d(x, pointwise_weights, strides = (1,1,1,1), padding = "SAME")
  x = tf.nn.relu6(x)
  return x

def model_forward(x):
  input_conv_weights = np.random.random((3, 3, 3, 8)).astype(np.float32)
  fc_weights = np.random.random((1, 1, 256, 1001)).astype(np.float32)
  x = tf.nn.conv2d(x, input_conv_weights, strides = (1,2,2,1), padding = "SAME")
  x = tf.nn.relu6(x)
  x = separable_conv(x, 8, 16, 1)
  x = separable_conv(x, 16, 32, 2)
  x = separable_conv(x, 32, 32, 1)
  x = separable_conv(x, 32, 64, 2)
  x = separable_conv(x, 64, 64, 1)
  x = separable_conv(x, 64, 128, 2)
  x = separable_conv(x, 128, 128, 1)
  x = separable_conv(x, 128, 128, 1)
  x = separable_conv(x, 128, 128, 1)
  x = separable_conv(x, 128, 128, 1)
  x = separable_conv(x, 128, 128, 1)
  x = separable_conv(x, 128, 256, 2)
  x = separable_conv(x, 256, 256, 1)
  x = tf.nn.avg_pool(x, 4, strides = (1,1,1,1), padding = "VALID")
  x = tf.nn.conv2d(x, fc_weights, strides = (1,1,1,1), padding = "SAME")
  x = tf.nn.relu6(x)
  x = tf.reshape(x, (1,-1))
  x = tf.nn.softmax(x)
  return x

f = tf.function(model_forward)

concrete_func = f.get_concrete_function(input_spec)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

