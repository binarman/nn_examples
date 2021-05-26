#!/usr/bin/python3
import numpy as np
import keras as K

# apply a 3x3 transposed convolution
# with stride 1x1 and 3 output filters on a 12x12 image:
model = K.Sequential()
model.add(Deconvolution2D(3, 3, 3, output_shape=(None, 3, 14, 14),
              border_mode='valid',
              input_shape=(3, 12, 12)))
# Note that you will have to change
# the output_shape depending on the backend used.

# we can predict with the model and print the shape of the array.
dummy_input = np.ones((32, 3, 12, 12))
# For TensorFlow dummy_input = np.ones((32, 12, 12, 3))
preds = model.predict(dummy_input)
print(preds.shape)
# Theano GPU: (None, 3, 13, 13)
# Theano CPU: (None, 3, 14, 14)
# TensorFlow: (None, 14, 14, 3)

exit()

# apply a 3x3 transposed convolution
# with stride 2x2 and 3 output filters on a 12x12 image:
model = Sequential()
model.add(Deconvolution2D(3, 3, 3, output_shape=(None, 3, 25, 25),
              subsample=(2, 2),
              border_mode='valid',
              input_shape=(3, 12, 12)))
model.summary()

# we can predict with the model and print the shape of the array.
dummy_input = np.ones((32, 3, 12, 12))
# For TensorFlow dummy_input = np.ones((32, 12, 12, 3))
preds = model.predict(dummy_input)
print(preds.shape)
# Theano GPU: (None, 3, 25, 25)
# Theano CPU: (None, 3, 25, 25)
# TensorFlow: (None, 25, 25, 3)
