#!/usr/bin/python3

import tensorflow as tf
import numpy as np

input_size = tf.constant([1,4,4,2])

# weights dimensions are W,H, in channels, out channels
input_data = np.ones([1,32,32,4], dtype = np.float32)

filters = np.array(np.random.uniform(low = -1., high = 1, size=[3, 3, 4, 1]), dtype=np.float32)
strides = (1, 2, 2, 1)
dilations = np.array((2, 2), dtype=np.int64)

f = tf.function(lambda x: tf.nn.depthwise_conv2d(x, filters, strides, "VALID", data_format="NHWC", dilations=dilations))

concrete_func = f.get_concrete_function(input_data)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

