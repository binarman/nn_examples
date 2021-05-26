#!/usr/bin/python3

import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

s=tf.compat.v1.Session()
s.as_default()

data_tensor = tf.compat.v1.placeholder(shape=(1,3,4,5), dtype=tf.float32)
pads_tensor = tf.compat.v1.placeholder(shape=(4,2), dtype=tf.int32)

pad = tf.compat.v1.pad(data_tensor, pads_tensor)

converter = tf.compat.v1.lite.TFLiteConverter.from_session(s, [data_tensor, pads_tensor], [pad])

tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

