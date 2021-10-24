#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import time

def forward(x):
  filters1 = np.ones((3,3,32,64), dtype=np.float32)
  filters2 = np.ones((3,3,64,128), dtype=np.float32)
  filters3 = np.ones((3,3,128,256), dtype=np.float32)
  x = tf.pad(x, [[0,0], [0,1], [0,1], [0, 0]])
  x = tf.nn.conv2d(x, filters1, [1, 2, 2, 1], "VALID")
  x = tf.pad(x, [[0,0], [0,1], [0,1], [0, 0]])
  x = tf.nn.conv2d(x, filters2, [1, 2, 2, 1], "VALID")
  x = tf.pad(x, [[0,0], [0,1], [0,1], [0, 0]])
  x = tf.nn.conv2d(x, filters3, [1, 2, 2, 1], "VALID")
  filters = np.ones((3,3,32,64), dtype=np.float32)
  return x

input_spec = tf.TensorSpec((1,128,128,32), dtype=tf.float32)
cf = tf.function(forward).get_concrete_function(input_spec)

converter = tf.lite.TFLiteConverter.from_concrete_functions([cf])
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
  f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_content = tflite_model)
interpreter.allocate_tensors()

start = time.time()
for i in range(1000):
  interpreter.invoke()
end = time.time()
print("elapsed: ", end - start)

