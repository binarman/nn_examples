#!/usr/bin/python3

import tensorflow as tf
import numpy as np
# Construct a basic model.
root = tf.train.Checkpoint()

root.f = tf.function(lambda x: tf.linalg.set_diag(x, np.array([[1,2]]*2, dtype=np.float32), k=1))

# Create the concrete function.
x = np.ones(shape=[2, 3, 4], dtype=np.float32)
concrete_func = root.f.get_concrete_function(x)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

