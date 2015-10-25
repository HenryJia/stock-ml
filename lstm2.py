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
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

import matplotlib.pyplot as plt

from get import get

n = 5
days = 10
batch_size = 10
learning_rate = 1.0
decay_rate = 1e-6
momentum = 0.95
lstm1_units = 256
lstm2_units = 512
lstm3_units = 1024
dense_units = 512
grad_clip = 100
random.seed(0)
epochs = 50

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
  y = np.zeros(length)
  for i in range(length):
    x[i] = data[i:(i + m)]
    y[i] = data[(i + m), 3]
    #y[i] = (data[(i + m), 3] - x_mean[3]) / x_std[3]
  return x, y

def read_model_data(model, filename):
  """Unpickles and loads parameters into a Lasagne model."""
  filename = os.path.join('./', '%s.%s' % (filename, param_extension))
  with open(filename, 'r') as f:
    data = cPickle.load(f)
  lasagne.layers.set_all_param_values(model, data)


def write_model_data(model, filename):
  """Pickels the parameters within a Lasagne model."""
  data = lasagne.layers.get_all_param_values(model)
  filename = os.path.join('./', filename)
  filename = '%s.%s' % (filename, param_extension)
  with open(filename, 'w') as f:
    cPickle.dump(data, f)


def validate(data_validate):
  pos_test = 0
  testx, testy = gen_data_next(data_validate, days, pos_test, data_validate.shape[0] - days - 1)
  overall_test_error = loss_fn(testx, testy)
  #print 'Validation error ', overall_test_error
  return overall_test_error

print 'Generating Data...'
data = get('stock_data_clean/KAZ_LDX.TXT', -1, 100)
data_test = get('stock_data_clean/KAZ_LDX.TXT', 100, 1)

data_np = np.array(data)
data_test_np = np.array(data_test)

data_np, mean_np, std_np = normalize(data_np)
data_test_np = normalize_known(data_test_np, mean_np, std_np)

x, y = gen_data(data_np, days)
x_test, y_test = gen_data(data_test_np, days)

print 'Building Model...'
model = Sequential()
model.add(LSTM(n, lstm1_units, return_sequences=True))
model.add(LSTM(lstm1_units, lstm2_units, return_sequences=True))
model.add(LSTM(lstm2_units, lstm3_units, return_sequences=False))
model.add(Dense(lstm3_units, dense_units))
model.add(Dense(dense_units, 1))
model.add(Activation("linear"))
#sgd = optimizers.SGD(lr=learning_rate, decay=decay_rate, momentum=momentum, nesterov=False)
#adagrad = optimizers.adagrad(lr=learning_rate)
adadelta = optimizers.Adadelta(lr=learning_rate)
print 'Compiling...'
model.compile(loss="mean_squared_error", optimizer='rmsprop')

#if os.path.isfile('keras-weights.nn'):
  #print 'Loading Model...'
  #model.load_weights('keras-weights.nn')
print 'Begin Training...'
model.fit(x, y, batch_size=batch_size, nb_epoch=epochs)#, validation_split=0.05)
print 'Testing Model...'
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print 'Test Score: ', score
print 'Test Error: ', (math.sqrt(score) * std_np[3])
print 'Saving Model...'
model.save_weights('keras-weights.nn', overwrite=True)
result = (model.predict(x_test, batch_size=batch_size, verbose=1) * std_np[3] + mean_np[3])
#for i in range(len(y_test)):
  #print 'Predictions: ', result[i] , ' Correct: ', (y_test[i] * std_np[3] + mean_np[3])
plt.plot(result, 'bs')
plt.plot((y_test * std_np[3] + mean_np[3]), 'r^')
plt.show()
print 'All Done :D'