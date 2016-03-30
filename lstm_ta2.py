import numpy as np
import random
import os
import time
import csv
import math
import sys

import theano
import theano.tensor as T

from keras.models import Sequential
from keras import optimizers
from keras.layers.core import TimeDistributedDense, Dropout, Activation, Reshape, Flatten
from keras.layers.recurrent import LSTM, SimpleRNN
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick2_ohlc

import talib as tl

from get import get_all

epsilon = 1

n = 5
batch_size = 100
days = 8
grad_clip = 100
random.seed(0)
epochs = 100

def normalize(data):
  mean = np.mean(data, axis=0)
  std = np.std(data, axis=0)
  result = (data - mean) / std
  return result, mean, std

def normalize_known(data, mean, std):
  result = (data - mean) / std
  return result
count = 0

def gen_data(data, target, m):
  forwards = 4
  length = data.shape[0] - m + 1 - forwards
  x = np.zeros((length, m, data.shape[1]))
  y = np.zeros((length, m, 1))
  y_dir = np.zeros(length)
  for i in range(length):
    x[i] = data[i:(i + m), :]
    y[i] = np.expand_dims(target[(i + forwards):(i + m + forwards)], axis = 1)
  return x, y

def remove_nan(data):
  return data[~np.isnan(data).any(axis=1)]

print 'Generating Data...'
Symbols = []
with open('symbols_clean.csv') as csvfile:
  reader = csv.DictReader(csvfile)
  Symbols.append(reader.fieldnames[0])
  for line in reader:
    for k in line.items():
      Symbols.append(k[1])

data_raw = []
for s in Symbols:
  data_raw += [np.array(get_all(('stock_data_clean/' + s)))]

data_all = []
target_all = []
data_test_all = []
target_test_all = []
for d in data_raw:
  indicator = []
  for i in xrange(d.shape[1]):
    indicator += [np.transpose(np.asarray(tl.MACD(d[:, i], float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))))]

  data_dirty = np.concatenate((indicator + [np.expand_dims(d[:, 3], axis = 1)]), axis = 1)
  data_clean = remove_nan(data_dirty)

  data = data_clean[:-200, :-1]
  target = data_clean[:-200, -1]

  data_test = data_clean[-200:, :-1]
  target_test = data_clean[-200:, -1]

  data, mean, std = normalize(data)
  data_test = normalize_known(data_test, mean, std)
  
  data_all += [data]
  target_all += [target]

  data_test_all += [data_test]
  target_test_all += [target_test]
print 'Data Loaded...'
wait = raw_input('Press To Continue')

print 'Building Model...'
model = Sequential()
#model.add(LSTM(100, stateful=True, return_sequences=True, activation='tanh', batch_input_shape=(batch_size, days, n)))
model.add(LSTM(100, return_sequences=True, activation='tanh', input_shape=(None, data_all[0].shape[1] )))
model.add(LSTM(100, return_sequences=True, activation='tanh'))
model.add(TimeDistributedDense(1, activation='linear'))

print 'Compiling...'
optimizer = optimizers.Adam(lr=0.005)
model.compile(loss="mean_squared_error", optimizer=optimizer)

#if os.path.isfile('keras-weights.nn'):
  #print 'Loading Model...'
  #model.load_weights('keras-weights.nn')
print 'Begin Training...'
for i in xrange(1, days):
  print 'Pretraining: Length: ', 2**i
  for e in xrange(epochs):
    for d, t in zip(data_all, target_all):
      x, y = gen_data(d, t, 2**i)
      model.fit(x, y, batch_size=batch_size, nb_epoch=1)#, validation_data=(x_test, y_test))

print 'Testing Model...'
for d, t in zip(data_test_all, target_test_all):
  score += model.evaluate(d, t, batch_size=batch_size, verbose=1)
print 'Test Score: ', score / len(data_test_all)

for e in xrange(epochs):
  for d, t in zip(data_all, target_all):
    x, y = gen_data(d, t, 200)
    x_test = np.expand_dims(d_test[:-1, :], axis = 0)
    y_test = np.expand_dims(np.expand_dims(t_test[1:], axis = 1), axis = 0)

    model.fit(x, y, batch_size=batch_size, nb_epoch=1)#, validation_data=(x_test, y_test))#, validation_split=0.05)
print 'Saving Model...'
model.save_weights('keras-weights.nn', overwrite=True)

print 'Testing Model...'
for d_test, t_test in zip(data_test_all, target_test_all):
  x_test = np.expand_dims(d_test[:-1, :], axis = 0)
  y_test = np.expand_dims(np.expand_dims(t_test[1:], axis = 1), axis = 0)
  score += model.evaluate(d, t, batch_size=batch_size, verbose=1)
print 'Test Score: ', score / len(data_test_all)

#result = model.predict(x_test[0], verbose=1)

plt.show()
print 'All Done :D'