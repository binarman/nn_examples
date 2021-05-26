#!/usr/bin/python3

import tensorflow as tf
import numpy as np
# Construct a basic model.
root = tf.train.Checkpoint()

root.f = tf.function(lambda x, y: tf.raw_ops.PadV2(input=x, paddings=y, constant_values=1))

# Create the concrete function.
x = np.ones(shape=[1, 1, 1, 1], dtype=np.float32)
paddings = np.array([[1,1], [2,2], [3,3], [4,4]], dtype=np.int32)
concrete_func = root.f.get_concrete_function(x, paddings)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

