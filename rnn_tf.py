import time

start = time.time()

import random
import os
import time
import csv
import math
import sys
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
from tensorflow.models.rnn import rnn
#from tensorflow.models.rnn.ptb import reader
from tensorflow.models.rnn import linear

from get import get

epsilon = 1

n = 5
batch_size = 100
train_days = 30
test_days = 30
lstm1_units = 32
lstm2_units = 32
lstm3_units = 32
dense_units = 512
grad_clip = 100
random.seed(0)
epochs = 100

Symbols = []

#Read the symbols list

with open('symbols_processed4.csv') as csvfile:
  reader = csv.DictReader(csvfile)
  Symbols.append(reader.fieldnames[0])
  for line in reader:
    for k in line.items():
      Symbols.append(k[1])

#data2 = get('stock_data_clean/GLEN_LDX.TXT', 1000, 100)
#data_test2 = get('stock_data_clean/GLEN_LDX.TXT', 200, 1)

#data_np2 = np.array(data2)
#data_test_np2 = np.array(data_test2)
def normalize(data):
  mean = np.mean(data, axis=0)
  std = np.std(data, axis=0)
  result = (data - mean) / std
  return result, mean, std

def normalize_known(data, mean, std):
  result = (data - mean) / std
  return result
count = 0
def get_next_extreme(data):
  if (data[0, 3] <= data[0 + 1, 3]):
    for i in range(len(data) - 1):
      if (data[i, 3] > data[i + 1, 3]):
        return data[i, :]
  elif (data[0, 3] > data[0 + 1, 3]):
    for i in range(len(data) - 1):
      if (data[i, 3] < data[i + 1, 3]):
        return data[i, :]
  global count
  count += 1
  print 'Default used ', count, ' times'
  return data[-1, :]

def gen_data(data, m, predict_date, extreme):
  length = data.shape[0] - m - predict_date
  x = np.zeros((length, m, 5), dtype=np.float32)
  y = np.zeros((length, m, 4), dtype=np.float32)
  y_dir = np.zeros(length, dtype=np.float32)
  for i in range(length):
    x[i] = data[i:(i + m)]
    if (extreme == True):
      for j in range(m):
        y[i, j, :] = get_next_extreme(data[(i + j):, 0:4])
    elif (extreme == False):
      y[i] = data[(i + predict_date):(i + m + predict_date), 0:4]
    if (data[(i + m), 3] > data[(i + m - predict_date), 3]):
        y_dir[i] = 1
    #y[i] = (data[(i + m), 3] - x_mean[3]) / x_std[3]
  return x, y, y_dir

print 'Generating Data...'
data = get('stock_data_clean/KAZ_LDX.TXT', -1, 200)
data_test = get('stock_data_clean/KAZ_LDX.TXT', 200, 1)

data_np = np.array(data)
data_test_np = np.array(data_test)

#for x in np.nditer(data_np):
  #if (x <= 0):
    #x = 1

#for x in np.nditer(data_test_np):
  #if (x <= 0):
    #x = 1
data_np[:, 4] += 1
data_test_np[:, 4] += 1

data_np = np.log(data_np)
data_test_np = np.log(data_test_np)

data_np, mean_np, std_np = normalize(data_np)
data_test_np = normalize_known(data_test_np, mean_np, std_np)

x, y, y_dir = gen_data(data_np, train_days, 0, True) # 3rd argument not ncesssary if true
x_test, y_test, y_test_dir = gen_data(data_test_np, test_days, 1, False)
print 'NaN Check x ', np.isnan(np.sum(x))
print 'NaN Check y ', np.isnan(np.sum(y))
if (np.isnan(np.sum(x)) or np.isnan(np.sum(y))):
  sys.exit('Error NaN(s) detected.')

print 'Building Model...'

class customRNNCell(rnn_cell.RNNCell):
  """The Output RNN Cell"""

  def __init__(self, num_units, input_size):
    self._num_units = num_units
    self._input_size = input_size

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Most basic RNN: output = new_state = tanh(W * input + U * state + B)."""
    with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
      output = tf.tanh(linear.linear([inputs, state], self._num_units, True))
    return output, output

def run_model(inputs, targets, time_frames):
  layers = []

  cell = customRNNCell(lstm1_units, 5)
  layers.append(rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5))
  cell = customRNNCell(lstm2_units, lstm1_units)
  layers.append(rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5))
  cell = customRNNCell(lstm3_units, lstm2_units)
  layers.append(rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5))

  layers.append(customRNNCell(4, lstm1_units))

  model = rnn_cell.MultiRNNCell(layers)

  initial_state = model.zero_state(batch_size, tf.float32)

  outputs = []
  states = []
  state = initial_state
  in_list = tf.split(1, time_frames, inputs)
  #target_list = tf.split(1, time_frames, targets))
  for time_step, input_ in enumerate(in_list):
    input_ = tf.squeeze(input_)
    if time_step > 0: tf.get_variable_scope().reuse_variables()
    (model_output, state) = model(input_, state)
    outputs.append(model_output)
    states.append(state)

  return outputs, states

train_data = tf.placeholder(tf.float32, shape=(batch_size, train_days, 5))
train_targets = tf.placeholder(tf.float32, shape=(batch_size, train_days, 4))

out, internal_states = run_model(train_data, train_targets, train_days)
for i in xrange(len(out)):
  out[i] = tf.reshape(out[i], shape=(batch_size, 1, 4))
output = tf.concat(1, out)#, [-1, size])
loss = tf.reduce_mean(tf.square(output - train_targets))

batch = tf.Variable(0)
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.0).minimize(loss, global_step=batch)

with tf.Session() as s:
  tf.initialize_all_variables().run()
  print('Initialized!')
  # Loop through training steps.
  for step in xrange(epochs * x.shape[0] // batch_size):
    offset = (step * batch_size) % (x.shape[0] - batch_size)
    batch_data = x[offset:(offset + batch_size), :, :]
    batch_targets = y[offset:(offset + batch_size), :, :]
    feed_dict = {train_data: batch_data, train_targets: batch_targets}
    _, l, predictions = s.run([optimizer, loss, output], feed_dict=feed_dict)
    if step % 100 == 0:
      print('Epoch %.2f' % (float(step) * batch_size / x.shape[0]))
      print('Minibatch loss: %.3f' % l)
      #print('Minibatch error: %.1f%%' % error_rate(predictions, batch_targets))
      #print('Validation error: %.1f%%' %
            #error_rate(validation_prediction.eval(), validation_labels))
      sys.stdout.flush()

end = time.time()
print end - start