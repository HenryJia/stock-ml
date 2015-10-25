import cPickle, gzip, numpy, random, os, time, math, csv, os
import lasagne
import theano
import theano.tensor as T

from get import get

Symbols = []

#Read the symbols list

with open('symbols_processed4.csv') as csvfile:
  reader = csv.DictReader(csvfile)
  Symbols.append(reader.fieldnames[0])
  for line in reader:
    for k in line.items():
      Symbols.append(k[1])

n = 5
days = 100
batch_size = 1
learning_rate = 10
momentum = 0.95
lstm_units = 1
dense_units = 100
grad_clip = 100
random.seed(0)
iterations = 200 * 50

data1 = get('KAZ_LDX.TXT', 2000, 100)
data2 = get('GLEN_LDX.TXT', 1000, 100)
data_test1 = get('KAZ_LDX.TXT', 200, 1)
data_test2 = get('GLEN_LDX.TXT', 200, 1)

data_np1 = numpy.array(data1)
data_test_np1 = numpy.array(data_test1)
data_np2 = numpy.array(data2)
data_test_np2 = numpy.array(data_test2)
#data_np = numpy.vstack((numpy.array(data1), numpy.array(data2)))
#data_test_np = numpy.vstack((numpy.array(data_test1), numpy.array(data_test2)))
#numpy.random.shuffle(data_np)
#numpy.random.shuffle(data_test_np)
#print data_np.shape
#print data_test_np.shape
#datax_np = data_np[0:199]
#datay_np = data_np[1:200, 3]

#input_var = T.matrix('X')
target_var = T.vector('y')

def gen_data_rand(data, m, bs = 1): # Randomly sample batch_size pieces of stock data of length m
  length = data.shape[0] - m - 1
  x = numpy.zeros((bs, m, 5))
  y = numpy.zeros(bs)
  for i in range(bs):
    start = random.randint(0, length)
    x[i] = data[start:(start + m)]
    y[i] = data[(start + m), 3]
  return x, y

def gen_data_next(data, m, pos, bs = 1):
  length = data.shape[0] - m - 1
  if (pos >= length):
    pos = 0
  real_bs = min(length - pos, bs)
  x = numpy.zeros((real_bs, m, 5))
  y = numpy.zeros(real_bs)
  for i in range(real_bs):
    x[i] = data[pos:(pos + m)]
    y[i] = data[(pos + m), 3]
    pos += 1
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

#l_input = lasagne.layers.InputLayer((None, days, 5))
l_input =lasagne.layers.InputLayer(shape=(None, days, 5))
l_lstm1 = lasagne.layers.LSTMLayer(l_input, lstm_units, grad_clipping=grad_clip, nonlinearity=lasagne.nonlinearities.tanh)
#l_lstm2 = lasagne.layers.LSTMLayer(l_lstm1, lstm_units, grad_clipping=grad_clip, nonlinearity=lasagne.nonlinearities.tanh)
#l_lstm3 = lasagne.layers.LSTMLayer(l_lstm2, lstm_units, grad_clipping=grad_clip, nonlinearity=lasagne.nonlinearities.tanh)
#l_slice = lasagne.layers.SliceLayer(l_lstm1, -1, 1)
#l_dense1 = lasagne.layers.DenseLayer(l_slice, num_units=dense_units, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.tanh)
l_out = lasagne.layers.DenseLayer(l_lstm1, num_units=1, W = lasagne.init.Normal(),  nonlinearity=lasagne.nonlinearities.linear)

#read_model_data(l_out, "network")

prediction = lasagne.layers.get_output(l_out)
loss = lasagne.objectives.squared_error(prediction.flatten(), target_var).mean()

params = lasagne.layers.get_all_params(l_out)
updates = lasagne.updates.adagrad(loss, params, learning_rate)
#updates = lasagne.updates.momentum(loss, params, learning_rate, momentum)

print 'compiling predict function'
predict_fn = theano.function([l_input.input_var], prediction, allow_input_downcast=True)
print 'compiling loss function'
loss_fn = theano.function([l_input.input_var, target_var], loss, allow_input_downcast=True)
print 'compiling update function'
train_fn = theano.function([l_input.input_var, target_var], loss, updates=updates, allow_input_downcast=True)
print 'all done'

pos_train1 = 0
pos_train2 = 0
start = time.time()
for i in range(350):
  x1, y1 = gen_data_next(data_np1, days, pos_train1, batch_size)
  #x2, y2 = gen_data_next(data_np2, days, pos_train2, batch_size)
  loss = train_fn(x1, y1)
  #loss += train_fn(x2, y2)
  validation = validate(data_test_np1)
  #validation += validate(data_test_np2)
  print 'Epoch', i, '  validation loss; ', validation, " Time: ", time.time() - start
  #print "Epoch ", i, ": ", loss, " Time: ", time.time() - start
  write_model_data(l_out, "network")
  #if(i % 1 == 0):


validate(data_test_np)

