#!/usr/bin/python3

import tensorflow as tf
import numpy as np
# Construct a basic model.
root = tf.train.Checkpoint()

#root.f = tf.function(lambda cond: tf.where(cond, [[1,2,3,4]], [100,200,300,400]))
root.f = tf.function(lambda cond: tf.where(cond)) 

#x = tf.constant(2)
#y = tf.constant(5)
#def f1(): return tf.multiply(x, 17)
#def f2(): return tf.add(y, 23)
#r = tf.cond(tf.less(x, y), tf.multiply(x, 2), tf.add(x, 3))

# Create the concrete function.
input_x = np.ones(shape=[4], dtype=np.bool)
concrete_func = root.f.get_concrete_function(input_x)

print("create concrete function")

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

