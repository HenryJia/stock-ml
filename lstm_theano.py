from collections import OrderedDict
import cPickle, gzip, random, os, time, math, csv, os
import numpy as np
import numpy.matlib
import theano
import theano.tensor as T

import matplotlib.pyplot as plt
from matplotlib.finance import candlestick2_ohlc

from get import get

random.seed(0)

epsilon = 1

n = 5
lstm1_units = 2048
lstm2_units = 2048
lstm3_units = 32
dense_units = 512
epochs = 500

#def orthogonal(shape, scale=1.1, name=None):
def orthogonal(shape, scale=0.9, name=None):
  ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
  '''
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape).astype('float32')
  u, _, v = np.linalg.svd(a, full_matrices=False)
  # pick the one with the correct shape
  q = u if u.shape == flat_shape else v
  q = q.reshape(shape)
  return theano.shared(value=scale * q[:shape[0], :shape[1]], name=name, strict=False)

def glorot_uniform(shape, name=None):
  fan_in, fan_out = shape[0], shape[1]
  s = np.sqrt(6. / (fan_in + fan_out))
  return theano.shared(value=np.random.uniform(low=-s, high=s, size=shape).astype('float32'), name=name, strict=False)

def one(shape, name=None):
  return theano.shared(value=np.ones(shape, dtype=np.float32), name=name, strict=False)

def zero(shape, name=None):
  return theano.shared(value=np.zeros(shape, dtype=np.float32), name=name, strict=False)

def linear(x):
  return x

class LSTM(object):
  state_len = 2

  def __init__(self, input_dim, output_dim, init=glorot_uniform, inner_init=orthogonal, forget_bias_init=one, activation=T.tanh, inner_activation=T.nnet.hard_sigmoid):
    self.output_dim = output_dim
    self.init = init
    self.inner_init = inner_init
    self.forget_bias_init = forget_bias_init
    self.activation = activation
    self.inner_activation = inner_activation

    self.input_dim = input_dim

    self.W_i = self.init((self.input_dim, self.output_dim))
    self.U_i = self.inner_init((self.output_dim, self.output_dim))
    self.b_i = zero((self.output_dim,))
    #self.b_i = theano.shared(value=np.zeros((self.output_dim,), dtype=np.float32))

    self.W_f = self.init((self.input_dim, self.output_dim))
    self.U_f = self.inner_init((self.output_dim, self.output_dim))
    self.b_f = self.forget_bias_init((self.output_dim,))

    self.W_c = self.init((self.input_dim, self.output_dim))
    self.U_c = self.inner_init((self.output_dim, self.output_dim))
    self.b_c = zero((self.output_dim,))
    #self.b_c = theano.shared(value=np.zeros((self.output_dim,), dtype=np.float32))

    self.W_o = self.init((self.input_dim, self.output_dim))
    self.U_o = self.inner_init((self.output_dim, self.output_dim))
    self.b_o = zero((self.output_dim,))
    #self.b_o = theano.shared(value=np.zeros((self.output_dim,), dtype=np.float32))

    self.params = [self.W_i, self.U_i, self.b_i,
                    self.W_c, self.U_c, self.b_c,
                    self.W_f, self.U_f, self.b_f,
                    self.W_o, self.U_o, self.b_o]

  def step(self, x, *states):
    #print x
    #print len(states)
    assert len(states) == 2
    h_tm1 = states[0]
    c_tm1 = states[1]

    x_i = T.dot(x, self.W_i) + self.b_i
    x_f = T.dot(x, self.W_f) + self.b_f
    x_c = T.dot(x, self.W_c) + self.b_c
    x_o = T.dot(x, self.W_o) + self.b_o

    i = self.inner_activation(x_i + T.dot(h_tm1, self.U_i))
    f = self.inner_activation(x_f + T.dot(h_tm1, self.U_f))
    c = f * c_tm1 + i * self.activation(x_c + T.dot(h_tm1, self.U_c))
    o = self.inner_activation(x_o + T.dot(h_tm1, self.U_o))
    h = o * self.activation(c)
    return [h, c]

class Dense(object):
  state_len = 0

  def __init__(self, input_dim, output_dim, init=glorot_uniform, activation=T.nnet.relu):
    self.init = init
    self.activation = activation
    self.input_dim = input_dim
    self.output_dim = output_dim

    self.W = self.init((self.input_dim, self.output_dim))
    self.b = zero((self.output_dim,))

    self.params = [self.W, self.b]

  def step(self, x, *states):
    output = self.activation(T.dot(x, self.W) + self.b)
    return [output]


class Adam(object):

  def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    self.iterations = theano.shared(np.float32(0))
    self.lr = theano.shared(np.float32(lr))
    self.beta_1 = theano.shared(np.float32(beta_1))
    self.beta_2 = theano.shared(np.float32(beta_2))
    self.epsilon = epsilon

  def get_updates(self, params, grads):
    self.updates = [(self.iterations, self.iterations+1.)]

    t = self.iterations + 1
    lr_t = self.lr * T.sqrt(1 - T.pow(self.beta_2, t)) / (1 - T.pow(self.beta_1, t))

    for p, g in zip(params, grads):
      # zero init of moment
      m = theano.shared(np.zeros(p.get_value().shape, dtype=np.float32))
      # zero init of velocity
      v = theano.shared(np.zeros(p.get_value().shape, dtype=np.float32))

      m_t = (self.beta_1 * m) + (1 - self.beta_1) * g
      v_t = (self.beta_2 * v) + (1 - self.beta_2) * T.square(g)
      p_t = p - lr_t * m_t / (T.sqrt(v_t) + self.epsilon)

      self.updates.append((m, m_t))
      self.updates.append((v, v_t))
      self.updates.append((p, p_t))
      return self.updates

def rnn_compile(layers, optimiser):
  initial_states = []
  params = []

  for l in layers:
    params += l.params
    for i in xrange(l.state_len):
      initial_states.append(T.zeros_like(zero(l.output_dim)))

  def rnn(x, *states):
    prev_out = x
    state_num = 0
    new_states = []
    for l in layers:
      output = l.step(prev_out, *states[state_num:state_num + l.state_len])
      if l.state_len > 0:
        new_states += output
      prev_out = output[0]
      state_num += l.state_len
    return [prev_out] + new_states

  data = T.matrix()
  targets = T.matrix()

  result, updates = theano.scan(fn=rnn, outputs_info=[None] + initial_states, sequences=data)
  run_fn = theano.function(inputs=[data], outputs=result)

  loss = T.mean((result[0] - targets) ** 2, axis=None)
  loss_fn = theano.function(inputs=[data, targets], outputs=loss)

  gradients = T.grad(loss, params)
  #param_updates = optimiser.get_updates(params, gradients)
  param_updates = OrderedDict((p, p - 0.01*g) for p, g in zip(params, gradients))
  train_fn = theano.function(inputs=[data, targets], outputs=loss, updates=param_updates)
  return run_fn, loss_fn, train_fn

layers = [LSTM(5, lstm1_units), Dense(lstm1_units, 5)]
adam = Adam(lr=0.1)
run_fn, loss_fn, train_fn = rnn_compile(layers, adam)

def normalize(data):
  mean = np.mean(data, axis=0)
  std = np.std(data, axis=0)
  result = (data - mean) / std
  return result, mean, std

def normalize_known(data, mean, std):
  result = (data - mean) / std
  return result

def recover(data, mean, std):
  result = data * std + mean
  return result

data_raw = get('stock_data_clean/RDSA_LDX.TXT', -1, 200)
data_test_raw = get('stock_data_clean/RDSA_LDX.TXT', 200, 1)

data, mu, sigma = normalize(np.array(data_raw, dtype=np.float32))
data_test = normalize_known(np.array(data_test_raw, dtype=np.float32), mu, sigma)

x = data[:-1, :]
y = data[1:, :]
x_test = data_test[:-1, :]
y_test = data_test[1:, :]

for i in xrange(epochs):
  training_loss = train_fn(x, y)
  test_loss = loss_fn(x_test, y_test)
  print "Epoch #", i, " Training Loss: ", training_loss, "; Test Loss: ", test_loss

result = run_fn(x[-100:])[0]

f, axarr = plt.subplots(2, sharey=True)
recovered_result = recover(result, mu, sigma)
candlestick2_ohlc(axarr[0], recovered_result[:, 0], recovered_result[:, 1], recovered_result[:, 2], recovered_result[:, 3], width=1, colorup='g', colordown='r')
#plt.plot(np.exp(result[:, -1, 3]), 'bs')
recovered_y = recover(y[-100:], mu, sigma)
candlestick2_ohlc(axarr[1], recovered_y[:, 0], recovered_y[:, 1], recovered_y[:, 2], recovered_y[:, 3], width=1, colorup='g', colordown='r')
#plt.plot(np.exp(y_test[:, -1, 3] * std_np[3] + mean_np[3]), 'r^')
#plt.plot(np.exp(y_test[:, -1, 3]), 'r^')
f, axarr = plt.subplots(2, sharey=True)
plt.show()
