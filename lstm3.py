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

from get import get

epsilon = 1

n = 5
batch_size = 1
days = 12
grad_clip = 100
random.seed(0)
epochs = 100

Symbols = []

def normalize(data):
  mean = np.mean(data, axis=0)
  std = np.std(data, axis=0)
  result = (data - mean) / std
  return result, mean, std

def normalize_known(data, mean, std):
  result = (data - mean) / std
  return result
count = 0

def gen_data(data, m):
  length = data.shape[0] - m - 1
  x = np.zeros((length, m, 5))
  y = np.zeros((length, m, 5))
  y_dir = np.zeros(length)
  for i in range(length):
    x[i] = data[i:(i + m)]
    y[i] = data[(i + 1):(i + m + 1)]
    if (data[(i + m), 3] > data[(i + m - 1), 3]):
        y_dir[i] = 1
    #y[i] = (data[(i + m), 3] - x_mean[3]) / x_std[3]
  return x, y, y_dir

print 'Generating Data...'
data_raw = get('stock_data_clean/KAZ_LDX.TXT', -1, 400)
data_test_raw = get('stock_data_clean/KAZ_LDX.TXT', 400, 1)

data = np.array(data_raw)
data_test = np.array(data_test_raw)

data, mean, std = normalize(data)
data_test = normalize_known(data_test, mean, std)

#x_test, y_test, y_test_dir = gen_data(data_test, days)
#print 'NaN Check x ', np.isnan(np.sum(x))
#print 'NaN Check y ', np.isnan(np.sum(y))
#if (np.isnan(np.sum(x)) or np.isnan(np.sum(y))):
  #sys.exit('Error NaN(s) detected.')

print 'Building Model...'
model = Sequential()
#model.add(LSTM(lstm1_units, stateful=True, return_sequences=True, activation='tanh', batch_input_shape=(batch_size, days, n)))
model.add(LSTM(lstm1_units, return_sequences=True, activation='tanh'))
model.add(TimeDistributedDense(5, activation='linear'))
sgd = optimizers.SGD(lr=0.1, decay=0, momentum=0.9, nesterov=True)
adagrad = optimizers.adagrad(lr=0.1)
adadelta = optimizers.Adadelta(lr=1.0)
print 'Compiling...'
model.compile(loss="mean_squared_error", optimizer='adam')

if os.path.isfile('keras-weights.nn'):
  print 'Loading Model...'
  model.load_weights('keras-weights.nn')
#print 'Begin Training...'
#for i in xrange(1, days):
  #print 'Pretraining: Length: ', 2**i
  #x, y, y_dir = gen_data(data, 2**i)
  #model.fit(x, y, batch_size=batch_size, nb_epoch=1)
data = np.reshape(data, (1, ) + data.shape)
data_test = np.reshape(data_test, (1, ) + data_test.shape)

print 'Testing Model...'
score = model.evaluate(data_test[:, :-1, :], data_test[:, 1:, :], batch_size=batch_size, verbose=1, show_accuracy=True)
print 'Test Score: ', score

#model.fit(data[:, :-1, :], data[:, 1:, :], batch_size=batch_size, nb_epoch=epochs, validation_data=(data_test[:, :-1, :], data_test[:, 1:, :]))#, validation_split=0.05)
#print 'Saving Model...'
#model.save_weights('keras-weights.nn', overwrite=True)

result = model.predict(data_test[:, :-1, :], verbose=1)
f, axarr = plt.subplots(2, sharey=True)
recovered_result = result[0] * std + mean
candlestick2_ohlc(axarr[0], recovered_result[:, 0], recovered_result[:, 1], recovered_result[:, 2], recovered_result[:, 3], width=1, colorup='g', colordown='r')
#plt.plot(np.exp(result[:, -1, 3]), 'bs')
recovered_data_test = data_test[0, 1:, :] * std + mean
candlestick2_ohlc(axarr[1], recovered_data_test[:, 0], recovered_data_test[:, 1], recovered_data_test[:, 2], recovered_data_test[:, 3], width=1, colorup='g', colordown='r')
#plt.plot(np.exp(y_test[:, -1, 3] * std[3] + mean[3]), 'r^')
#plt.plot(np.exp(y_test[:, -1, 3]), 'r^')

plt.show()
print 'All Done :D'