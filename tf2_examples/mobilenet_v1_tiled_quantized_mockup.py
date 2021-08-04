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

def separable_conv_tiled(x, input_channels, output_channels, stride):
  depthwise_weights = (np.random.random((3, 3, input_channels, 1)) * 2 - 1).astype(np.float32)
  pointwise_weights = (np.random.random((1, 1, input_channels, output_channels)) * 2 - 1).astype(np.float32)
  x = tf.nn.depthwise_conv2d(x, depthwise_weights, strides = (1,stride,stride,1), padding = "VALID")
  x = tf.nn.relu6(x)
  x = tf.nn.conv2d(x, pointwise_weights, strides = (1,1,1,1), padding = "VALID")
  x = tf.nn.relu6(x)
  return x

def get_idx_image(input_size, output_size, filter_size, strides, idx):
  padding = output_size*strides - input_size
  image_begin = idx * strides - padding // 2
  return (image_begin, image_begin + filter_size - 1)

def get_tile_part1_image(low, high):
  img1_low = get_idx_image(64, 32, 3, 2, low)
  img2_low = get_idx_image(64, 64, 3, 1, img1_low[0])
  img3_low = get_idx_image(128, 64, 3, 2, img2_low[0])

  img1_high = get_idx_image(64, 32, 3, 2, high)
  img2_high = get_idx_image(64, 64, 3, 1, img1_high[1])
  img3_high = get_idx_image(128, 64, 3, 2, img2_high[1])
  return (img3_low[0], img3_high[1])

def tiled_part_1(x, w_tiles, h_tiles):
  input_conv_weights = (np.random.random((3, 3, 3, 8)) * 2 - 1).astype(np.float32)

  tile = []
  h_input = 128
  w_input = 128
  h_output = 32
  w_output = 32
  h_tile_size = h_output//h_tiles
  w_tile_size = w_output//w_tiles
  input_conv_weights = (np.random.random((3, 3, 3, 8)) * 2 - 1).astype(np.float32)

  input_channels_1 = 8
  output_channels_1 = 16
  stride_1 = 1
  depthwise_weights_1 = (np.random.random((3, 3, input_channels_1, 1)) * 2 - 1).astype(np.float32)
  pointwise_weights_1 = (np.random.random((1, 1, input_channels_1, output_channels_1)) * 2 - 1).astype(np.float32)

  input_channels_2 = 16
  output_channels_2 = 32
  stride_2 = 2
  depthwise_weights_2 = (np.random.random((3, 3, input_channels_2, 1)) * 2 - 1).astype(np.float32)
  pointwise_weights_2 = (np.random.random((1, 1, input_channels_2, output_channels_2)) * 2 - 1).astype(np.float32)

  for i in range(h_tiles):
    tile += [[]]
    for j in range(w_tiles):

      tile_top = i * h_tile_size
      tile_bottom = (i+1) * h_tile_size - 1
      tile_left = j *w_tile_size
      tile_right = (j+1) * w_tile_size - 1

      if i == h_tiles - 1:
        tile_bottom += h_output % h_tile_size

      if j == w_tiles - 1:
        tile_right += w_output % w_tile_size

      # inference origin of output tile
      left_bound, right_bound = get_tile_part1_image(tile_left, tile_right)
      top_bound, bottom_bound = get_tile_part1_image(tile_top, tile_bottom)

      left_pad = 0
      right_pad = 0
      top_pad = 0
      bottom_pad = 0

      if left_bound < 0:
        left_pad = -left_bound
        left_bound = 0
      if right_bound >= w_input:
        right_pad = right_bound - w_input + 1
        right_bound = w_input - 1

      if top_bound < 0:
        top_pad = - top_bound
        top_bound = 0
      if bottom_bound >= h_input:
        bottom_pad = bottom_bound - h_input + 1
        bottom_bound = h_input - 1

      y = tf.slice(x, (0, top_bound, left_bound, 0), (1, bottom_bound - top_bound + 1, right_bound - left_bound + 1, 3))

      need_padding = left_pad != 0 or right_pad != 0 or top_pad != 0 or bottom_pad != 0
      if need_padding:
        y = tf.pad(y, [[0,0],[top_pad, bottom_pad],[left_pad, right_pad],[0,0]])

      y = tf.nn.conv2d(y, input_conv_weights, strides = (1,2,2,1), padding = "VALID")
      y = tf.nn.relu6(y)

      y = tf.nn.depthwise_conv2d(y, depthwise_weights_1, strides = (1,stride_1,stride_1,1), padding = "VALID")
      y = tf.nn.relu6(y)

      y = tf.nn.conv2d(y, pointwise_weights_1, strides = (1,1,1,1), padding = "VALID")
      y = tf.nn.relu6(y)

      y = tf.nn.depthwise_conv2d(y, depthwise_weights_2, strides = (1,stride_2,stride_2,1), padding = "VALID")
      y = tf.nn.relu6(y)
      tile[i] += [y]
  
  hor_concats = []
  for i in range(h_tiles):
    hor_concats += [tf.concat(tile[i], axis = 2)]
  x = tf.concat(hor_concats, axis = 1)

  x = tf.nn.conv2d(x, pointwise_weights_2, strides = (1,1,1,1), padding = "SAME")
  x = tf.nn.relu6(x)
  return x

def model_forward(x):
  x = tiled_part_1(x, 4, 4)
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
  fc_weights = (np.random.random((1, 1, 256, 1001)) * 2 - 1).astype(np.float32)
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

#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.representative_dataset = representative_dataset
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.int8
#converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

