#!/usr/bin/python3

import numpy as np
import tensorflow as tf

i = tf.keras.Input(shape=[2,8], batch_size=1)
o = tf.keras.layers.GRU(4, unroll=True)(i)
m = tf.keras.Model(i, o)

c = tf.lite.TFLiteConverter.from_keras_model(m)
tflite = c.convert()

with open("gru.tflite", "wb") as f:
  f.write(tflite)

