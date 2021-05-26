#!/usr/bin/python3

import tensorflow as tf
import tensorflow.keras as K

model = K.Sequential()
#model.add(K.layers.Bidirectional(K.layers.LSTM(10, return_sequences=True), input_shape=(5, 10)))
model.add(K.layers.Bidirectional(K.layers.SimpleRNN(10), input_shape = (5, 10)))
model.add(K.layers.Dense(5))
model.add(K.layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

 # With custom backward layer
# model = Sequential()
# forward_layer = LSTM(10, return_sequences=True)
# backward_layer = LSTM(10, activation='relu', return_sequences=True,
#                       go_backwards=True)
# model.add(Bidirectional(forward_layer, backward_layer=backward_layer,
#                         input_shape=(5, 10)))
# model.add(Dense(5))
# model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
