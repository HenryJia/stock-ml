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

def make_model():
  print 'Building Model...'
  model = Sequential()
  #model.add(LSTM(100, stateful=True, return_sequences=True, activation='tanh', batch_input_shape=(batch_size, 1, x_all.shape[2])))
  model.add(SimpleRNN(100, return_sequences=True, activation='tanh', input_shape=(None, 5)))
  model.add(SimpleRNN(100, return_sequences=True, activation='tanh'))
  model.add(TimeDistributedDense(1, activation='linear'))
  print 'Compiling...'
  optimizer = optimizers.Adam(lr=0.001)
  model.compile(loss="mean_squared_error", optimizer=optimizer)
  return model

def run(data_raw, model):
  #data_raw += [np.array(get_all(('stock_data_clean/' + s)))]
  data = data_raw[~np.any(data_raw == 0, axis=1)]
  print data.shape, data_raw.shape
  std = np.load('std4.npy')
  mean = np.load('mean4.npy')
  print np.std(((data[1:, :] / data[:-1, :] - 1)[-200:, :] - mean) / std, axis = 0)
  x = np.expand_dims(((data[1:, :] / data[:-1, :] - 1)[-200:, :] - mean) / std, axis = 0)
  print x.shape
  result = model.predict(x, verbose=1)
  print 'All Done :D'
  return result

#run(rio, model)[-1, -1, -1]
