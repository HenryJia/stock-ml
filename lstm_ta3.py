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

batch_size = 100
days = 8
grad_clip = 100
random.seed(0)
epochs = 100

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

print 'Generating Data...'
Symbols = []
with open('symbols_processed4.csv') as csvfile:
  reader = csv.DictReader(csvfile)
  Symbols.append(reader.fieldnames[0])
  for line in reader:
    for k in line.items():
      Symbols.append(k[1])

data_raw = np.array(get_all(('stock_data_clean/KAZ_LDX.TXT')))

data_all = add_all_indicators(data_raw)

data = data_all[:-201, :-1]
target = data_all[:-201, -1]

data_test = data_all[-201:, :-1]
target_test = data_all[-201:, -1]

data, data_test, mean, std = normalize(data, data_test)

print data.shape
x_all, y_all = gen_data(data, target, 200)
x_test = np.expand_dims(data_test[:-4, :], axis = 0)
y_test = np.expand_dims(np.expand_dims(target_test[4:], axis = 1), axis = 0)
print x_all.shape, y_all.shape, x_test.shape, y_test.shape

print 'Building Model...'
model = Sequential()
#model.add(LSTM(100, stateful=True, return_sequences=True, activation='tanh', batch_input_shape=(batch_size, 1, x_all.shape[2])))
model.add(LSTM(100, return_sequences=True, activation='tanh', input_shape=(None, x_all.shape[2])))
model.add(LSTM(100, return_sequences=True, activation='tanh'))
model.add(TimeDistributedDense(1, activation='linear'))

print 'Compiling...'
optimizer = optimizers.Adam(lr=0.005)
model.compile(loss="mean_squared_error", optimizer=optimizer)

if os.path.isfile('keras-weights.nn'):
  print 'Loading Model...'
  model.load_weights('keras-weights.nn')
print 'Begin Training...'
for i in xrange(1, days):
  print 'Pretraining: Length: ', 2**i
  x, y = gen_data(data, target, 2**i)
  model.fit(x, y, batch_size=batch_size, nb_epoch=epochs)#, validation_data=(x_test, y_test))

print 'Testing Model...'
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print 'Test Score: ', score

model.fit(x_all, y_all, batch_size=batch_size, nb_epoch=epochs, validation_data=(x_test, y_test))#, validation_split=0.05)
print 'Saving Model...'
model.save_weights('keras-weights.nn', overwrite=True)

print 'Testing Model...'
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print 'Test Score: ', score

result = model.predict(x_test, verbose=1)

plt.show()
print 'All Done :D'