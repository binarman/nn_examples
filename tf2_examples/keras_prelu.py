#!/usr/bin/python3
import numpy as np
import tensorflow as tf
import tensorflow.keras as K

# apply a 3x3 transposed convolution
# with stride 1x1 and 3 output filters on a 12x12 image:
model = K.Sequential()
model.add(K.layers.PReLU())
model.build((3,3))
model.save("permute.h5")
# Note that you will have to change
# the output_shape depending on the backend used.

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tfmodel)

