#!/usr/bin/python3

import tensorflow as tf
import tensorflow.keras.layers as layers

model = tf.keras.models.Sequential()
model.add(layers.Conv2D(8, [3, 3], strides=(2,2), padding='same', activation='relu')) # 32->16
model.add(layers.Conv2D(16, [3, 3], strides=(2,2), padding='same', activation='relu')) # 16->8
model.add(layers.Conv2D(32, [3, 3], strides=(2,2), padding='same', activation='relu')) # 8->4
model.add(layers.Flatten())
model.add(layers.Dense(11, activation='softmax'))

model.build([1, 32, 32, 1])

c = tf.lite.TFLiteConverter.from_keras_model(model)
tflite = c.convert()

with open("model.tflite", "wb") as f:
  f.write(tflite)

