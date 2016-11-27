import theano
import theano.tensor as T

from keras.models import Sequential
from keras import optimizers
from keras.layers import TimeDistributed, Dense, Dropout, Activation, Reshape, Flatten
from keras.layers import LSTM, SimpleRNN

def deep_lstm():
    print 'Building Model...'
    model = Sequential()
    model.add(SimpleRNN(200, return_sequences=True, activation='tanh', input_shape=(None, 5)))
    model.add(SimpleRNN(100, return_sequences=True, activation='tanh'))
    #model.add(LSTM(100, return_sequences=True, activation='tanh'))
    model.add(TimeDistributed(Dense(4, activation='linear')))
    print 'Compiling...'
    model.compile(loss="mse", optimizer='adam', sample_weight_mode="temporal")
    return model
