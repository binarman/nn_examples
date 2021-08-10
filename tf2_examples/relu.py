#!/usr/bin/python3

import tensorflow as tf
import numpy as np
# Construct a basic model.

f = tf.function(lambda x: tf.nn.relu(x))

# Create the concrete function.
x = tf.TensorSpec(shape=[1, 3], dtype=np.float32)
concrete_func = f.get_concrete_function(x)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

