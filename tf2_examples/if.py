#!/usr/bin/python3

import tensorflow as tf
import numpy as np
# Construct a basic model.

#f = tf.function(lambda x,y: tf.realdiv(x, y))
def f1(x): return tf.multiply(x, 2)
def f2(x): return tf.add(x, 3)
f = tf.function(lambda cond,x: tf.cond(cond, lambda : f1(x), lambda : f2(x)))

#x = tf.constant(2)
#y = tf.constant(5)
#def f1(): return tf.multiply(x, 17)
#def f2(): return tf.add(y, 23)
#r = tf.cond(tf.less(x, y), tf.multiply(x, 2), tf.add(x, 3))

# Create the concrete function.
input_x = np.ones(shape=[3], dtype=np.float32)
cond = tf.constant(1)
concrete_func = f.get_concrete_function(cond, input_x)

print("create concrete function")

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

