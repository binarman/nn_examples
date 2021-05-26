#!/usr/bin/python

import tensorflow as tf
import numpy as np

s=tf.Session()
s.as_default()

data_tensor = tf.placeholder(shape=(1,3,4,5), dtype=tf.float32)
pads_tensor = tf.placeholder(shape=(4,2), dtype=tf.int32)
constants = tf.constant(0.0, shape=(), dtype=tf.float32)

pad = tf.pad(data_tensor, pads_tensor, mode="CONSTANT", constant_values=constants)

converter = tf.lite.TFLiteConverter.from_session(s, [data_tensor, pads_tensor], [pad])

tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

