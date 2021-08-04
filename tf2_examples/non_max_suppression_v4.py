#!/usr/bin/python3

import tensorflow as tf
import numpy as np
# Construct a basic model.

#f = tf.function(lambda boxes, scores, max_output_size: tf.raw_ops.NonMaxSuppressionV4(boxes = boxes, scores = scores, max_output_size = max_output_size, iou_threshold=0.1, score_threshold=float('-100')))
f = tf.function(lambda boxes, scores, max_output_size: tf.image.non_max_suppression_padded(boxes = boxes, scores = scores, max_output_size = max_output_size, iou_threshold=0.1, score_threshold=float('-100'), pad_to_max_output_size=False))

# Create the concrete function.
n_boxes = 10
boxes = np.ones(shape=[n_boxes, 4], dtype=np.float32)
scores = np.ones(shape=[n_boxes], dtype=np.float32)
max_output_size = n_boxes // 2

concrete_func = f.get_concrete_function(boxes, scores, max_output_size)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

