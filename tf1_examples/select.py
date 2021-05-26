#!/usr/bin/python

import tensorflow as tf
import numpy as np

s=tf.Session()
s.as_default()

cond = tf.placeholder(shape=(3,4), dtype=tf.bool)
t = tf.placeholder(shape=(3,4), dtype=tf.float32)
f = tf.placeholder(shape=(3,4), dtype=tf.float32)

select = tf.where(cond, t, f)

converter = tf.lite.TFLiteConverter.from_session(s, [cond, t, f], [select])

tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

