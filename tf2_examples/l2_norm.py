#!/usr/bin/python3

import tensorflow as tf
import numpy as np
# Construct a basic model.
root = tf.train.Checkpoint()

root.f = tf.function(lambda x: tf.nn.relu(tf.math.l2_normalize(x)))

# Create the concrete function.
input_data = tf.constant(1., shape=[1, 4, 4, 1])
concrete_func = root.f.get_concrete_function(input_data)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open("1model.tflite", "wb") as f:
    f.write(tflite_model)

