#!/usr/bin/python3

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

print(tfa.__version__)
print(tf.__version__)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Input(shape=(32,32,1), name="input_layer"))
model.add(tfa.layers.InstanceNormalization(axis=3,
                                           beta_initializer="random_uniform",
                                           gamma_initializer="random_uniform"))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.allow_custom_ops = True
tflite_model = converter.convert()
tflite_file = 'model.tflite'
open(tflite_file, "wb").write(tflite_model)
