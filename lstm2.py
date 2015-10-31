import numpy as np
import random
import os
import time
import csv
import math

import theano
import theano.tensor as T

from keras import optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt

from get import get

epsilon = 1

n = 5
days = 20
batch_size = 10
lstm1_units = 256
lstm2_units = 512
lstm3_units = 1
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
data = get('stock_data_clean/KAZ_LDX.TXT', -1, 100)
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

#data_np, mean_np, std_np = normalize(data_np)
#data_test_np = normalize_known(data_test_np, mean_np, std_np)

x, y, y_dir = gen_data(data_np, days)
x_test, y_test, y_test_dir = gen_data(data_test_np, days)

print 'Building Model...'
model = Sequential()
#model.add(Reshape((1, days, n), input_shape=(days, n)))
#model.add(Convolution2D(64, 5, 5))
#model.add(Reshape((days - 5 + 1, 64)))
model.add(LSTM(lstm1_units, return_sequences=True, activation='tanh', input_shape=(days, n)))
model.add(Dropout(0.5))
model.add(LSTM(lstm2_units, return_sequences=True))
model.add(Dropout(0.5))
#model.add(LSTM(lstm3_units, return_sequences=True))
#model.add(Dropout(0.5))
model.add(SimpleRNN(5, return_sequences=True, activation='linear'))
sgd = optimizers.SGD(lr=0.1, decay=0, momentum=0.9, nesterov=True)
adagrad = optimizers.adagrad(lr=0.1)
adadelta = optimizers.Adadelta(lr=1.0)
print 'Compiling...'
model.compile(loss="mean_squared_error", optimizer=adadelta)

#if os.path.isfile('keras-weights.nn'):
  #print 'Loading Model...'
  #model.load_weights('keras-weights.nn')
print 'Testing Model...'
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
print 'Test Score: ', score
#print 'Test Error: ', (math.sq
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
plt.plot(np.exp(result[:, -1, 3]), 'bs')
#plt.plot(np.exp(y_test[:, 3] * std_np[3] + mean_np[3]), 'r^')
plt.plot(np.exp(y_test[:, -1, 3]), 'r^')
plt.show()
print 'All Done :D'