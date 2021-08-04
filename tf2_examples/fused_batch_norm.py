#!/usr/bin/python3

import tensorflow as tf
import numpy as np
# Construct a basic model.

mean = tf.constant([1., 2., 3.])
variance = tf.constant([4., 5., 6.])
#offset = tf.constant([10., 11., 12.])
#scale = tf.constant([7., 8., 9.])

#f = tf.function(lambda x: tf.raw_ops.FusedBatchNorm(x=x, scale=scale, offset=offset, mean=mean, variance=variance, epsilon=1e-5))
f = tf.function(lambda x:tf.compat.v1.nn.fused_batch_norm(x=x, scale=None, offset=None, mean=mean, variance=variance, epsilon=1e-5, is_training=False))

# Create the concrete function.
input_data = tf.constant(1., shape=[1, 3, 3, 3])
concrete_func = f.get_concrete_function(input_data)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

converter.allow_custom_ops = True

tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

