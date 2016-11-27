import numpy as np
import random
import os
import time
import csv
import math
import sys
from glob import glob

import theano
import theano.tensor as T

import matplotlib.pyplot as plt
from matplotlib.finance import candlestick2_ohlc

#import talib as ta
from models_lib import deep_lstm

batch_size = 20
days = 8
final_length = 256
random.seed(0)
epochs = 1
epochs_full = 5

look_ahead = 1

def preprocess(data_raw):
    data = data_raw[~np.any(data_raw == 0, axis=1)]
    result = data[1:, :] / data[:-1, :] - 1
    result = result[:-look_ahead, :]
    #result_target = np.zeros((result.shape[0], 1))
    #for i in xrange(result.shape[0]):
        #result_target[i] = data[i + 1 + look_ahead, 3] / data[i + 1, 3] - 1
    result_target = data[look_ahead + 1:, :4] / data[1:-look_ahead, :4] - 1
    #return np.concatenate([result, result_target], axis = 1)
    return result, result_target

def gen_data(data, target, m):
    length = data.shape[0] - m + 1
    x = np.zeros((length, m, data.shape[1]))
    y = np.zeros((length, m, 4))
    y_dir = np.zeros(length)
    for i in range(length):
        x[i] = data[i:(i + m), :]
        y[i] = target[i:(i + m), :]
    return x, y

data_raw = []
directory = '../data_npy/'
print glob('../data_npy/'  + '*.npy')
for fn in glob('../data_npy/'  + '*.npy'):
    data_raw += [np.load(fn)]

#data_all = add_all_indicators(data_raw)
data_all = []
target_all = []
for d in data_raw:
    d_p, t_p = preprocess(d)
    data_all += [d_p]
    target_all += [t_p]

data = []
target = []
data_test = []
target_test = []
for d, t in zip(data_all, target_all):
    data += [d[:-200]]
    target += [t[:-200]]

    data_test += [d[-200:]]
    target_test += [t[-200:]]

x_all = []
y_all = []
x_test = []
y_test = []
data = data[0:1]
target = target[0:1]
data_test = data_test[0:1]
target_test = target_test[0:1]
for d, t, d_t, t_t in zip(data, target, data_test, target_test):
    if d.shape[0] < final_length:
        continue
    x, y = gen_data(d, t, final_length)
    x_all += [x]
    y_all += [y]
    x_test += [np.expand_dims(d_t, axis = 0)]
    y_test += [np.expand_dims(t_t, axis = 0)]

x_all = np.concatenate(x_all, axis = 0)
y_all = np.concatenate(y_all, axis = 0)

x_test = np.concatenate(x_test, axis = 0)
y_test = np.concatenate(y_test, axis = 0)

model = deep_lstm()

#if os.path.isfile('keras-weights4.nn'):
    #print 'Loading Model...'
    #model.load_weights('keras-weights4.nn')

print 'Begin Training...'
#for i in xrange(1, days):
    #print 'Pretraining: Length: ', 2**i
    #print 'Generating Data'
    #x_pre = []
    #y_pre = []
    #for d, t in zip(data, target):
        #if d.shape[0] < 200:
            #continue
        #x, y = gen_data(d, t, 2**i)
        #x_pre += [x]
        #y_pre += [y]

    #x_pre = np.concatenate(x_pre, axis = 0)
    #y_pre = np.concatenate(y_pre, axis = 0)

    #print 'Pretraining ', epochs, 'epochs'
    #model.fit(x_pre, y_pre, batch_size=batch_size, nb_epoch=epochs, validation_data=(x_test, y_test), shuffle = True)

print 'Testing Model...'
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print 'Test Score: ', score

model.fit(x_all, y_all, batch_size=batch_size, nb_epoch=epochs_full, validation_data=(x_test, y_test), shuffle = True)#, validation_split=0.05)
#print 'Saving Model...'
#model.save_weights('keras-weights.nn', overwrite=True)

print 'Testing Model...'
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print 'Test Score: ', score

result = model.predict(x_test, verbose=1)

print 'All Done :D'

output = model.predict(x_test)

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

capital, position = PnL(output[0, :, 3], y_test[0, :, 3])

length = y_test.shape[1]
#length = 100

plt.figure(1)
plt.plot(capital[:length])

data_test_orig = data_raw[0][-200:]
plt.figure(2)
ax1 = plt.subplot(211)
plt.plot(data_test_orig[:length + 1, 3])
plt.subplot(212, sharex = ax1)
plt.scatter(np.arange(length + 1), position[:length + 1])
plt.show(block = False)
