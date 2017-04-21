from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import codecs
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import time
import argparse


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

sys.argv = ['test_rnn']

parser = argparse.ArgumentParser(description='LSTM text generation.')
parser.add_argument('--data', dest='data', type=str, help='data file')
parser.add_argument('--prefix', dest='prefix', type=str, help='output file prefix')
parser.add_argument('--vec', dest='vec', type=str, help='w2v model file')
parser.add_argument('--maxlen', dest='maxlen', type=int, help='sequence length')
parser.add_argument('--step', dest='step', type=int, help='sequence generated with step')
parser.add_argument('--layers', dest='layers', type=int, help='number of layers')
parser.add_argument('--lsize', dest='lsize', type=int, help='size of each layer')
parser.add_argument('--gen_length', dest='gen_length', type=int, help='length of text generated after each batch')
parser.add_argument('--batch', dest='batch', type=int, help='batch size in chars as read from file')
parser.add_argument('--model_batch', dest='model_batch', type=int, help='batch size in model training')
parser.add_argument('--dropout', dest='dropout', type=float, help='dropout prob between LSTM layers')
parser.add_argument('--gpu_fraction', dest='gpu_fraction', type=float, help='GPU memory fraction to be used by TensorFlow')
parser.set_defaults(
    data='/media/data/wiki.processed', prefix='model.lstm',
    lsize=200, layers=1, step=25, maxlen=100, dropout=0.2, batch=1000, model_batch=256,
    gen_length=200, gpu_fraction=0.8,
)
args = parser.parse_args()


def get_session(gpu_fraction=0.3):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session(args.gpu_fraction))

chars = []
char_indices = {}
indices_char = {}
unic_chars = set()
cities = []
maxlen = 0

with open('../data/nlp/US_Cities.txt') as f:
    for line in f:
        line = line.strip()
        cities.append(line)
        if len(line) > maxlen:
            maxlen = len(line)

        for letter in line:
            unic_chars.add(letter)

unic_chars.add('#')
unic_chars.add('!')

i = 0
for c in unic_chars:
    chars.append(c)
    char_indices[c] = i
    indices_char[i] = c
    i += 1
chars = set(chars)
print('total chars:', len(chars))

maxlen = 7

print('Build model...')
model = Sequential()

model.add(LSTM(args.lsize, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(args.dropout))
for _ in range(args.layers - 1):
    model.add(LSTM(args.lsize, return_sequences=True, input_shape=(maxlen, len(chars))))
    model.add(Dropout(args.dropout))
model.add(LSTM(args.lsize, return_sequences=False))

# model.add(LSTM(maxlen, return_sequences=False, input_shape=(maxlen, len(chars))))
model.add(Dropout(args.dropout))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
json_string = model.to_json()
open('%s.json' % args.prefix, 'w').write(json_string)

writefile = open('result.txt', 'a')


for iteration in range(1, 3000):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    cities = random.sample(cities, len(cities))
    text = '#'.join(cities)

    city_num = 0
    X = np.zeros((args.batch, maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((args.batch, len(chars)), dtype=np.bool)

    for i in range(len(text)-1):
        x_chars = []
        y_chars = []

        x_chars = list(text[i:i + maxlen])
        y_chars = list(text[i + 1:i + maxlen + 1])

        for t, c in enumerate(x_chars):
            X[city_num, t, char_indices[c]] = 1
        y[city_num, char_indices[y_chars[-1]]] = 1

        city_num += 1
        if city_num == args.batch:
            model.fit(X, y, batch_size=args.batch, nb_epoch=1)
            # model.save_weights('%s.i%d.h5' % (args.prefix, iteration), overwrite=True)
            city_num = 0
            X = np.zeros((args.batch, maxlen, len(chars)), dtype=np.bool)
            y = np.zeros((args.batch, len(chars)), dtype=np.bool)

    random_city = 'New York#'
    for i in range(30):

        X = np.zeros((1, maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((1, len(chars)), dtype=np.bool)

        x_chars = list(random_city[-maxlen-1:-1])
        y_chars = list(random_city[-maxlen:])

        for t, c in enumerate(x_chars):
            X[0, t, char_indices[c]] = 1
        y[0, char_indices[y_chars[-1]]] = 1

        preds = model.predict(X, verbose=0)[0]
        if sum(preds) > 1:
            continue
        next_index = sample(preds, 0.2)
        next_char = indices_char[next_index]
        random_city += next_char
    writefile.write('Iteration #' + str(iteration) + ' : ' + random_city + '\n')
    writefile.flush()
    #time.sleep(1)
