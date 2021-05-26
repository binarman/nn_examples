#!/usr/bin/python

import tensorflow as tf
import numpy as np
import sys

model_path = "model.tflite"
if len(sys.argv) > 1:
  model_path = sys.argv[1]

interpreter = tf.lite.Interpreter(model_path)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

for input_data in input_details:
  input_shape = input_data["shape"]
  input_dtype = input_data["dtype"]
  input_index = input_data["index"]

  input_data = np.ones(input_shape, input_dtype)
  interpreter.set_tensor(input_index, input_data)

interpreter.invoke()

output_details = interpreter.get_output_details()

output_tensor = interpreter.get_tensor(output_details[0]["index"])

print(output_tensor)

