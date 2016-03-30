import numpy as np
import random
import os
import time
import csv
import math
import sys

import theano
import theano.tensor as T

from keras import optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick2_ohlc

from get import get

epsilon = 1

n = 5
batch_size = 100
train_days = 150
test_days = 150
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
  x = np.zeros((length, m, 5))
  y = np.zeros((length, m, 4))
  y_dir = np.zeros(length)
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

x, y, y_dir = gen_data(data_np, train_days, 1, False) # 3rd argument not ncesssary if true
x_test, y_test, y_test_dir = gen_data(data_test_np, test_days, 1, False)
print 'NaN Check x ', np.isnan(np.sum(x))
print 'NaN Check y ', np.isnan(np.sum(y))
if (np.isnan(np.sum(x)) or np.isnan(np.sum(y))):
  sys.exit('Error NaN(s) detected.')

print 'Building Model...'
model = Sequential()
#model.add(Reshape((1, days, n), input_shape=(days, n)))
#model.add(Convolution2D(64, 5, 5))
#model.add(Reshape((days - 5 + 1, 64)))
model.add(LSTM(lstm1_units, return_sequences=True, activation='tanh', input_shape=(train_days, n)))
model.add(Dropout(0.5))
model.add(LSTM(lstm2_units, return_sequences=True, activation='tanh'))
model.add(Dropout(0.5))
model.add(LSTM(lstm3_units, return_sequences=True, activation='tanh'))
model.add(Dropout(0.5))
model.add(SimpleRNN(4, return_sequences=True, activation='linear'))
sgd = optimizers.SGD(lr=0.1, decay=0, momentum=0.9, nesterov=True)
adagrad = optimizers.adagrad(lr=0.1)
adadelta = optimizers.Adadelta(lr=1.0)
print 'Compiling...'
model.compile(loss="mean_squared_error", optimizer='adam')

#if os.path.isfile('keras-weights.nn'):
  #print 'Loading Model...'
  #model.load_weights('keras-weights.nn')
print 'Testing Model...'
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
print 'Test Score: ', score
print 'Begin Training...'
model.fit(x, y, batch_size=batch_size, nb_epoch=epochs, validation_data=(x_test, y_test))#, validation_split=0.05)
print 'Saving Model...'
model.save_weights('keras-weights.nn', overwrite=True)
#result = (model.predict(x_test, batch_size=batch_size, verbose=1) * std_np + mean_np)
result = model.predict(x_test, batch_size=batch_size, verbose=1)
print 'Testing Model...'
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
print 'Test Score: ', score
#print 'Test Error: ', (math.sqrt(score) * std_np[3])
result_dir = np.zeros(result.shape[0])
for i in range(len(result)):
  if (result[i, -1, 3] > x_test[i, -1, 3]):
    result_dir[i] = 1


wrong = 0
for i in range(len(result)):
  if (result_dir[i] != y_test_dir[i]):
    wrong += 1

percentage = wrong / float(len(result)) * 100
print 'Test Direction Error: ', wrong, ' Out of ', len(result),  ' Percentage: ', percentage
#for i in range(len(y_test)):
  #print 'Predictions: ', result[i] , ' Correct: ', (y_test[i] * std_np[3] + mean_np[3])
#plt.plot(np.exp(result[:, -1, 3] * std_np[3] + mean_np[3]), 'bs')
f, axarr = plt.subplots(2, sharey=True)
recovered_result = np.exp(result[:, -1, :] * std_np[0:4] + mean_np[0:4])
candlestick2_ohlc(axarr[0], recovered_result[:, 0], recovered_result[:, 1], recovered_result[:, 2], recovered_result[:, 3], width=1, colorup='g', colordown='r')
#plt.plot(np.exp(result[:, -1, 3]), 'bs')
recovered_y_test = np.exp(y_test[:, -1, :] * std_np[0:4] + mean_np[0:4])
candlestick2_ohlc(axarr[1], recovered_y_test[:, 0], recovered_y_test[:, 1], recovered_y_test[:, 2], recovered_y_test[:, 3], width=1, colorup='g', colordown='r')
#plt.plot(np.exp(y_test[:, -1, 3] * std_np[3] + mean_np[3]), 'r^')
#plt.plot(np.exp(y_test[:, -1, 3]), 'r^')
f, axarr = plt.subplots(2, sharey=True)
result_x = model.predict(x, batch_size=batch_size, verbose=1)
recovered_result_x = np.exp(result_x[-100:, -1, :] * std_np[0:4] + mean_np[0:4])

#result_dir_x = np.zeros(result_x.shape[0])
#for i in range(len(result_x)):
  #if (result_x[i, -1, 3] > x[i, -1, 3]):
    #result_dir_x[i] = 1


#wrong = 0
#for i in range(len(result_x)):
  #if (result_dir_x[i] != y_dir[i]):
    #wrong += 1

#percentage = wrong / float(len(result_dir_x)) * 100
#print 'Train Direction Error: ', wrong, ' Out of ', len(result_dir_x),  ' Percentage: ', percentage


#candlestick2_ohlc(axarr[0], recovered_result_x[:, 0], recovered_result_x[:, 1], recovered_result_x[:, 2], recovered_result_x[:, 3], width=1, colorup='g', colordown='r')
#recovered_y = np.exp(y[-100:, -1, :] * std_np[0:4] + mean_np[0:4])
#candlestick2_ohlc(axarr[1], recovered_y[:, 0], recovered_y[:, 1], recovered_y[:, 2], recovered_y[:, 3], width=1, colorup='g', colordown='r')
plt.show()
print 'All Done :D'