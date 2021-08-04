#!/usr/bin/python3

import tensorflow as tf
import numpy as np
# Construct a basic model.

diagonal = np.array([[[1, 2]*50]], dtype=np.float32)

#f = tf.function(lambda : tf.linalg.diag(diagonal, k=0, num_rows=2000, num_cols=2000))
#f = tf.function(lambda x : tf.linalg.diag(x, k=0, num_rows=100, num_cols=100))
f = tf.function(lambda x : tf.linalg.diag(x))

# Create the concrete function.
concrete_func = f.get_concrete_function(diagonal)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

