import tensorflow as tf
import numpy as np

tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

num_units = 1
lstm = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units = num_units)
batch=1
timesteps = 1
num_input = 1
X = tf.compat.v1.placeholder("float", [batch, timesteps, num_input])
x = tf.unstack(X, timesteps, 1)
outputs, states = tf.compat.v1.nn.static_rnn(lstm, x, dtype=tf.float32)

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, x, outputs)
tflite_model = converter.convert()
open("LSTM.tflite", "wb").write(tflite_model)

