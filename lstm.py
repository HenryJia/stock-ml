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
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick2_ohlc

n = 5
batch_size = 20
days = 8
grad_clip = 100
random.seed(0)
epochs = 30

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
  length = data.shape[0] - m + 1 - 1
  x = np.zeros((length, m, 5))
  y = np.zeros((length, m, 4))
  y_dir = np.zeros(length)
  for i in range(length):
    x[i] = data[i:(i + m)]
    y[i] = data[(i + 1):(i + m + 1), 0:4]
    if (data[(i + m), 3] > data[(i + m - 1), 3]):
        y_dir[i] = 1
  return x, y, y_dir

print 'Generating Data...'
data = np.genfromtxt('GOOGL_train20040101-20141231.csv', delimiter=',')[:, 1:]
data_test = np.genfromtxt('GOOGL_test20150101-20151231.csv', delimiter=',')[:, 1:]
data_test_orig = data_test

data = data[1:] / data[:-1] - 1
data_test = data_test[1:] / data_test[:-1] - 1

print 'Building Model...'
model = Sequential() # Edit the NN architecture here
model.add(LSTM(50, return_sequences=True, activation='tanh', input_shape=(None, 5)))
#model.add(LSTM(500, return_sequences=True, activation='tanh'))
#model.add(LSTM(500, return_sequences=True, activation='tanh'))
model.add(TimeDistributedDense(4, activation='linear'))

print 'Compiling...'
model.compile(loss="mean_squared_error", optimizer='adam')

#if os.path.isfile('keras-weights.nn'):
  #print 'Loading Model...'
  #model.load_weights('keras-weights.nn')
print 'Begin Training...'
#for i in xrange(1, days):
  #print 'Pretraining: Length: ', 2**i
  #x, y, y_dir = gen_data(data, 2**i)
  #model.fit(x, y, batch_size=batch_size, nb_epoch=1, shuffle=False)

data_test = np.reshape(data_test, (1, ) + data_test.shape)
x,y, y_dir = gen_data(data, 256)

print 'Testing Model...'
score = model.evaluate(data_test[:, :-1, :], data_test[:, 1:, :4], batch_size=batch_size, verbose=1)
print 'Test Score: ', np.sqrt(score)

model.fit(x, y, batch_size=batch_size, nb_epoch=epochs, shuffle=False, validation_data=(data_test[:, :-1, :], data_test[:, 1:, :4]))
print 'Saving Model...'
model.save_weights('keras-weights.nn', overwrite=True)

print 'Testing Model...'
score = model.evaluate(data_test[:, :-1, :], data_test[:, 1:, :4], batch_size=batch_size, verbose=1)
print 'Test Score: ', np.sqrt(score)

print 'All Done :D'

### New stuff for Shell Trading

output = model.predict(data_test[:, :-1, :])

def PnL(predicted, actual):
    assert predicted.shape[0] == actual.shape[0], 'Predicted and actual must be same length'
    capital = [1] # Suppose we invest according to the model and start with 1 pound
    position = [False] # Are we invested in it or not?
    for i in xrange(predicted.shape[0]):
        if predicted[i] > 0:
            position += [True]
        else:
            position += [False]
        if position[-1] == True:
            capital += [capital[-1] * (actual[i] + 1)]
        else:
            capital += [capital[-1]]
    return capital, position

capital, position = PnL(output[0, :, 3], data_test[0, 1:, 3])

#length = data_test.shape[1] - 1
length = 100

plt.figure(1)
plt.plot(capital[:length])
plt.figure(2)
ax1 = plt.subplot(211)
plt.plot(data_test_orig[:length + 1, 3])
plt.subplot(212, sharex = ax1)
plt.scatter(np.arange(length + 1), position[:length + 1])
plt.show(block = False)
