#!/usr/bin/python3

import tensorflow as tf
import numpy as np

# weights dimensions are W,H, in channels, out channels
input_spec = tf.TensorSpec([1, 128, 128, 3], dtype = np.float32)

def separable_conv(x, input_channels, output_channels, stride):
  depthwise_weights = (np.random.random((3, 3, input_channels, 1)) * 2 - 1).astype(np.float32)
  pointwise_weights = (np.random.random((1, 1, input_channels, output_channels)) * 2 - 1).astype(np.float32)
  x = tf.nn.depthwise_conv2d(x, depthwise_weights, strides = (1,stride,stride,1), padding = "SAME")
  x = tf.nn.relu6(x)
  x = tf.nn.conv2d(x, pointwise_weights, strides = (1,1,1,1), padding = "SAME")
  x = tf.nn.relu6(x)
  return x

def tiled_separable_conv_x2(x, tile_num):
  input_channels = 8
  output_channels = 16
  stride = 1
  depthwise_weights_1 = (np.random.random((3, 3, input_channels, 1)) * 2 - 1).astype(np.float32)
  pointwise_weights_1 = (np.random.random((1, 1, input_channels, output_channels)) * 2 - 1).astype(np.float32)
  x = tf.nn.depthwise_conv2d(x, depthwise_weights_1, strides = (1,stride,stride,1), padding = "SAME")
  x = tf.nn.relu6(x)

  tile = []
  processed_tile = []

  tile_size = 64//tile_num
  for i in range(tile_num):
    tile += [[]]
    for j in range(tile_num):
      tile[i] += [tf.slice(x, (0, tile_size * i, tile_size * j, 0), (1, tile_size, tile_size, input_channels))]
  print(tile)
  
  for i in range(tile_num):
    processed_tile += [[]]
    for j in range(tile_num):
      tmp = tf.nn.conv2d(tile[i][j], pointwise_weights_1, strides = (1,1,1,1), padding = "SAME")
      processed_tile[i] += [tf.nn.relu6(tmp)]
  print(processed_tile)

  input_channels = 16
  output_channels = 32
  stride = 2
  depthwise_weights_2 = (np.random.random((3, 3, input_channels, 1)) * 2 - 1).astype(np.float32)
  pointwise_weights_2 = (np.random.random((1, 1, input_channels, output_channels)) * 2 - 1).astype(np.float32)
  for i in range(tile_num):
    for j in range(tile_num):
      tmp = tf.nn.depthwise_conv2d(processed_tile[i][j], depthwise_weights_2, strides = (1,stride,stride,1), padding = "SAME")
      processed_tile[i][j] = tf.nn.relu6(tmp)
  
  hor_concats = []
  for i in range(tile_num):
    hor_concats += [tf.concat(processed_tile[i], axis = 2)]
  x = tf.concat(hor_concats, axis = 1)

  x = tf.nn.conv2d(x, pointwise_weights_2, strides = (1,1,1,1), padding = "SAME")
  x = tf.nn.relu6(x)
  return x


def model_forward(x):
  input_conv_weights = (np.random.random((3, 3, 3, 8)) * 2 - 1).astype(np.float32)
  fc_weights = (np.random.random((1, 1, 256, 1001)) * 2 - 1).astype(np.float32)
  x = tf.nn.conv2d(x, input_conv_weights, strides = (1,2,2,1), padding = "SAME")
  x = tf.nn.relu6(x)
  x = tiled_separable_conv_x2(x, 4)
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

def representative_dataset():
    for _ in range(1000):
      data = np.random.rand(1, 128, 128, 3) * 2 - 1
      yield [data.astype(np.float32)]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

