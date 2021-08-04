#!/usr/bin/python3

import tensorflow as tf
import numpy as np
# Construct a basic model.

f = tf.function(lambda cond, t, f: tf.compat.v1.where(cond, t, f))

# Create the concrete function.
cond = np.ones(shape=[2], dtype=np.bool)
t = np.ones(shape=[2,2,3,4], dtype=np.float32)
f = np.ones(shape=[2,2,3,4], dtype=np.float32)
concrete_func = f.get_concrete_function(cond, t, f)

print("create concrete function")

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

