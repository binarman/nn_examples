#!/usr/bin/python3

import tensorflow as tf
import numpy as np
# Construct a basic model.
root = tf.train.Checkpoint()

root.f = tf.function(lambda x: tf.compat.v1.math.round(x))

# Create the concrete function.
input_x = np.ones(shape=[3], dtype=np.float32)
concrete_func = root.f.get_concrete_function(input_x)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

converter.allow_custom_ops = True

tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

