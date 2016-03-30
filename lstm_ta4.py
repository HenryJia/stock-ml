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

#import talib as ta

from get import get_all
from indicator import add_all_indicators

batch_size = pow(2,14)
days = 8
grad_clip = 100
random.seed(0)
epochs = 10

def normalize(data, data_test):
  mean = np.mean(data, axis=0)
  std = np.std(data, axis=0)
  result = (data - mean) / std
  result_test = (data_test - mean) / std

  for i in xrange(std.shape[0]):
    if std[i] < 1e-5: # Use a small value instead of 0 to avoid numerical stability issues
      result[:, i] = 0
      result_test[:, i] = 0

  return result, result_test, mean, std

def normalize_known(data, mean, std):
  result = (data - mean) / std
  return result

def normalize_returns(data_list):
  data = np.concatenate(data_list, axis = 0)[:, :-1]
  print data.shape
  mean = np.mean(data, axis = 0)
  std = np.std(data, axis=0)

  result_list = data_list
  for i in xrange(len(data_list)):
    result_list[i][:, :-1] = (data_list[i][:, :-1] - mean) / std

  return result_list, mean, std

def preprocess(data_raw):
  data = data_raw[~np.any(data_raw == 0, axis=1)]
  result = data[1:, :] / data[:-1, :] - 1
  result = result[:-4, :]
  result_target = np.zeros((result.shape[0], 1))
  for i in xrange(result.shape[0]):
    result_target[i] = data[i + 1 + 4, 3] / data[i + 1, 3] - 1

  return np.concatenate([result, result_target], axis = 1)

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

def gen_data2(data, target, m):
  length = data.shape[0] - m + 1
  x = np.zeros((length, m, data.shape[1]))
  y = np.zeros((length, m, 1))
  y_dir = np.zeros(length)
  for i in range(length):
    x[i] = data[i:(i + m), :]
    y[i] = np.expand_dims(target[i:(i + m)], axis = 1)
  return x, y

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
  #data_raw += [np.array(get_all(('daily/' + s)))]

#data_all = add_all_indicators(data_raw)
data_all = []
for d in data_raw:
  data_all += [preprocess(d)]
  #data_all += [preprocess(d[:, 1:])]


data_all, mean, std = normalize_returns(data_all)

data = []
target = []
data_test = []
target_test = []
for d in data_all:
  data += [d[:-200, :-1]]
  target += [d[:-200, -1]]

  data_test += [d[-200:, :-1]]
  target_test += [d[-200:, -1]]

#data, data_test, mean, std = normalize(data, data_test)

#print data.shape
#x_all, y_all = gen_data(data, target, 200)
#x_test = np.expand_dims(data_test[:-4, :], axis = 0)
#y_test = np.expand_dims(np.expand_dims(target_test[4:], axis = 1), axis = 0)
#print x_all.shape, y_all.shape, x_test.shape, y_test.shape
x_all = []
y_all = []
x_test = []
y_test = []
for d, t, d_t, t_t in zip(data, target, data_test, target_test):
  x, y = gen_data2(d, t, 200)
  x_all += [x]
  y_all += [y]
  x_test += [np.expand_dims(d_t, axis = 0)]
  y_test += [np.expand_dims(np.expand_dims(t_t, axis = 1), axis = 0)]

x_all = np.concatenate(x_all, axis = 0)
y_all = np.concatenate(y_all, axis = 0)

x_test = np.concatenate(x_test, axis = 0)
y_test = np.concatenate(y_test, axis = 0)

print 'Building Model...'
model = Sequential()
#model.add(LSTM(100, stateful=True, return_sequences=True, activation='tanh', batch_input_shape=(batch_size, 1, x_all.shape[2])))
model.add(SimpleRNN(100, return_sequences=True, activation='tanh', input_shape=(None, x_all.shape[2])))
model.add(SimpleRNN(100, return_sequences=True, activation='tanh'))
model.add(TimeDistributedDense(1, activation='linear'))

print 'Compiling...'
optimizer = optimizers.Adam(lr=0.001)
model.compile(loss="mean_squared_error", optimizer=optimizer)

if os.path.isfile('keras-weights4.nn'):
  print 'Loading Model...'
  model.load_weights('keras-weights4.nn')


#print 'Begin Training...'
#for i in xrange(1, days):
  #print 'Pretraining: Length: ', 2**i
  #print 'Generating Data'
  #x_pre = []
  #y_pre = []
  #for d, t in zip(data, target):
    #x, y = gen_data(d, t, 2**i)
    #x_pre += [x]
    #y_pre += [y]

  #x_pre = np.concatenate(x_pre, axis = 0)
  #y_pre = np.concatenate(y_pre, axis = 0)

  #print 'Pretraining ', epochs, 's'
  #model.fit(x_pre, y_pre, batch_size=batch_size, nb_epoch=epochs, validation_data=(x_test, y_test))
  #batch_size  /= 2

#print 'Testing Model...'
#score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
#print 'Test Score: ', score

#model.fit(x_all, y_all, batch_size=batch_size, nb_epoch=epochs, validation_data=(x_test, y_test))#, validation_split=0.05)
#print 'Saving Model...'
#model.save_weights('keras-weights.nn', overwrite=True)

print 'Testing Model...'
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print 'Test Score: ', score

result = model.predict(x_test, verbose=1)

print 'All Done :D'
