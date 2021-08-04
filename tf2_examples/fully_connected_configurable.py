#!/usr/bin/python3
import sys

if len(sys.argv) != 3:
  print("usage: ./matmul_arbitrary_size.py <number of inputs> <number of outputs>")
  exit()

M, N = 1, 1

try:
  M, N = int(sys.argv[1]), int(sys.argv[2])
except:
  print("arguments should be integer")
  exit()

import tensorflow as tf
import numpy as np

# Construct a basic model.

# Create the concrete function.
weights = np.random.random((M, N)).astype(np.float32)
input_data = np.ones(shape=[1, M], dtype=np.float32)

f = tf.function(lambda x: tf.linalg.matmul(x, weights, transpose_a=False, transpose_b=False))

concrete_func = f.get_concrete_function(input_data)

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.allow_custom_ops = True

tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

