''' End-to-End memory network based on 
    https://github.com/fchollet/keras/blob/master/examples/babi_memnn.py
    Adding:
    - BoW encoding
    - Multiple hops
    - Adjacent embedding layers
 
'''

from __future__ import print_function
from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Merge, Dropout, RepeatVector, Lambda, Permute, Activation, Masking
from keras.layers import recurrent, Input, merge
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.optimizers import SGD, Adam
from keras.activations import softmax
from keras.metrics import categorical_accuracy
import keras.backend as K
from keras.utils.visualize_util import plot
import theano.tensor as T
import theano

from keras import initializations, regularizers, constraints

from functools import reduce
import tarfile
import numpy as np
#np.random.seed(1337)  # for reproducibility

import re
import pdb

from itertools import izip_longest

from os import listdir
from os.path import isfile, join

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)


class SequenceEmbedding(Embedding):
    def __init__(self, input_dim, output_dim, position_encoding=False, **kwargs):
        self.position_encoding = position_encoding
        self.zeros_vector =  T.zeros(output_dim, dtype='float32').reshape((1,output_dim))
        super(SequenceEmbedding, self).__init__(input_dim, output_dim, **kwargs)
    
 
    def call(self, x, mask=None):
        if 0. < self.dropout < 1.:
            retain_p = 1. - self.dropout
            B = K.random_binomial((self.input_dim,), p=retain_p) * (1. / retain_p)
            B = K.expand_dims(B)
            W = K.in_train_phase(self.W * B, self.W)
        else:
            W = self.W
        W_ = T.concatenate([self.zeros_vector, W], axis=0)
        out = K.gather(W_, x)
        return out

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q[:-1], a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append([sent[:-1]])
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data

def vectorize_facts(data, word_idx, story_maxlen, query_maxlen, fact_maxlen, enable_time = False):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = np.zeros((len(story), fact_maxlen),dtype='int32')
        for k,facts in enumerate(story):
            if not enable_time:
                x[k][-len(facts):] = np.array([word_idx[w] for w in facts])[:fact_maxlen]
            else:
                x[k][-len(facts)-1:-1] = np.array([word_idx[w] for w in facts])[:facts_maxlen-1]
                x[k][-1] = len(word_idx) + len(story) - k
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1) if not enable_time else np.zeros(len(word_idx) + 1 + story_maxlen)
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)

np.random.seed(1234)
#random.seed(1234)

'''
try:
    path = get_file('babi-tasks-v1-2.tar.gz', origin='http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
'''
tar = tarfile.open('babi-tasks-v1-2.tar.gz')

challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt',
    'three_supporting_facts_10k': 'tasks_1-20_v1-2/en/qa3_three-supporting-facts_{}.txt',

}
#challenge_type = 'single_supporting_fact_10k'
#challenge = challenges[challenge_type]
# Reading all file names
mypath = 'tasks_1-20_v1-2/en'
challenge_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
challenge_files = ['tasks_1-20_v1-2/en-10k/' + f.replace('train', '{}') for f 
                   in challenge_files if 'train.txt' == f[-9:]]

# Read all files
train_facts_split = []
test_facts_split = []
train_facts = []
test_facts = []
for challenge in challenge_files:
    train_facts_split.append(get_stories(tar.extractfile(challenge.format('train'))))
    test_facts_split.append(get_stories(tar.extractfile(challenge.format('test'))))
    train_facts += train_facts_split[-1]
    test_facts += test_facts_split[-1]

test_facts = np.array(test_facts)
train_facts = np.array(train_facts)
test_facts = list(test_facts[np.random.choice(len(test_facts), len(test_facts), replace=False)])
train_facts = list(train_facts[np.random.choice(len(train_facts), len(train_facts), replace=False)])


EMBED_HIDDEN_SIZE = 20
enable_time = True

#print('Extracting stories for the challenge:', challenge_type)
#train_facts = get_stories(tar.extractfile(challenge.format('train')))
#test_facts = get_stories(tar.extractfile(challenge.format('test')))

train_stories = [(reduce(lambda x,y: x + y, map(list,fact)),q,a) for fact,q,a in train_facts]
test_stories = [(reduce(lambda x,y: x + y, map(list,fact)),q,a) for fact,q,a in test_facts]

facts_maxlen = max(map(len, (x for h,_,_ in train_facts + test_facts for x in h)))
if enable_time:
    facts_maxlen += 1

story_maxlen = max(map(len, (x for x, _, _ in train_facts + test_facts)))
query_maxlen = max(map(len, (x for _, x, _ in train_facts + test_facts)))

vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train_stories + test_stories)))
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')

#facts_maxlen = query_maxlen = max(facts_maxlen, query_maxlen)
story_maxlen = 20 

if enable_time:
    vocab_size += story_maxlen

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_facts(train_facts, word_idx, story_maxlen, query_maxlen, facts_maxlen,
                                                               enable_time=enable_time)
inputs_test, queries_test, answers_test = vectorize_facts(test_facts, word_idx, story_maxlen, query_maxlen, facts_maxlen,
                                                         enable_time=enable_time)

print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('-')
print('Compiling...')

# This is a trick to avoid batch size average of error in training, 
# such as is done in Weston's code with size_average = False
class ModifiedBackprop(object):

    def __init__(self, nonlinearity):
        self.nonlinearity = nonlinearity
        self.ops = {}  # memoizes an OpFromGraph instance per tensor type

    def __call__(self, x):
        if theano.sandbox.cuda.cuda_enabled:
            maybe_to_gpu = theano.sandbox.cuda.as_cuda_ndarray_variable
        else:
            maybe_to_gpu = lambda x: x
        x = maybe_to_gpu(x)
        tensor_type = x.type
        if tensor_type not in self.ops:
            inp = tensor_type()
            outp = maybe_to_gpu(self.nonlinearity(inp))
            op = theano.OpFromGraph([inp], [outp])
            op.grad = self.grad
            self.ops[tensor_type] = op
        return self.ops[tensor_type](x)

softmax_grad = theano.tensor.nnet.nnet.SoftmaxGrad()
softmax_op = theano.tensor.nnet.nnet.Softmax()

class GuidedBackprop(ModifiedBackprop):
    def grad(self, inp, grads):
        x, = inp
        g_sm, = grads
        g_sm = g_sm * g_sm.shape[0].astype('float32') # Here I multiply to account for batch size division in
                                                        # keras
        sm = softmax_op(x)
        grad = [softmax_grad(g_sm, sm)]
        return grad

mod_softmax = GuidedBackprop(softmax)

print('Build model...')

def hop_layer(x, u, adjacent=None, input_layer=False, use_softmax=True):
    '''
        Define one hop of the memory network
    '''
    if adjacent == None:
        layer_encoder_m = SequenceEmbedding(input_dim=vocab_size-1,
                               output_dim=EMBED_HIDDEN_SIZE,
                               input_length=story_maxlen, init='normal')
    else:
        layer_encoder_m = adjacent
    

    #output: (samples, max_len, embedding_size )    
    # Memory
    input_encoder_m = layer_encoder_m(x)
        
    input_encoder_m = Lambda(lambda x: K.sum(x, axis=2),
                                 output_shape=(story_maxlen, EMBED_HIDDEN_SIZE,))(input_encoder_m)
    #input_encoder_m = Dropout(0.3)(input_encoder_m)
    
    memory = merge([input_encoder_m, u],
                    mode='dot',
                    dot_axes=[2, 1])
    if use_softmax:
        layer_memory = Lambda(lambda x: K.softmax(x))
    else:
        layer_memory = Lambda(lambda x: x)
    memory = layer_memory(memory)
    # output: (samples, max_len)

    # Output
    layer_encoder_c = SequenceEmbedding(input_dim=vocab_size-1,
                               output_dim=EMBED_HIDDEN_SIZE,
                               input_length=story_maxlen, init='normal')
    
    input_encoder_c = layer_encoder_c(x)
    input_encoder_c = Lambda(lambda x: K.sum(x, axis=2),
                                 output_shape=(story_maxlen, EMBED_HIDDEN_SIZE,))(input_encoder_c)
    #input_encoder_c = LSTM(EMBED_HIDDEN_SIZE, return_sequences=True)(input_encoder_c) # this gives 1.0 acc 
    #input_encoder_c = Dropout(0.3)(input_encoder_c)
    
    output = merge([memory, input_encoder_c],
                  mode = 'dot',
                  dot_axes=[1,1])
    output = merge([output, u], mode='sum')
    # output: (samples, embedding_size)
    layers = [layer_encoder_m, layer_encoder_c, layer_memory]
    return output, layers

# 2 hop memn2n
fact_input = Input(shape=(story_maxlen, facts_maxlen, ), dtype='int32', name='facts_input')
question_input = Input(shape=(query_maxlen, ), dtype='int32', name='query_input')

# input_length is different to input, so is not clear if I can share it, however this still 
# work because embedding layer does not check that
question_layer = SequenceEmbedding(input_dim=vocab_size-1,
                               output_dim=EMBED_HIDDEN_SIZE,
                               input_length=query_maxlen, init='normal')

question_encoder = question_layer(question_input)
#question_encoder = Dropout(0.3)(question_encoder)
question_encoder = Lambda(lambda x: K.sum(x, axis=1),
                         output_shape=lambda shape: (shape[0],) + shape[2:])(question_encoder)

#input_layer = Embedding(input_dim=vocab_size,
#                               output_dim=EMBED_HIDDEN_SIZE,
#                               input_length=story_maxlen, init='normal')
#input_layer.W = question_layer.W
#input_layer.trainable_weights = [input_layer.W]

o1,layers1 = hop_layer(fact_input, question_encoder, adjacent=question_layer, input_layer=True,
                      use_softmax=False)
o2,layers2 = hop_layer(fact_input, o1, adjacent=layers1[1], use_softmax=False)
o3,layers3 = hop_layer(fact_input, o2, adjacent=layers2[1], use_softmax=False)

# Response
response = Dense(vocab_size, init='normal', activation=mod_softmax, bias=False)(o3)
# This is not the way to do it
# TODO: Share softmax weights with input weights
#response.W = layers3[1].W.T
#response.trainable_weights = [response.W]

model = Model(input=[fact_input, question_input], output=[response])

#theano.printing.pydotprint(response, outfile="model.png", var_with_name_simple=True)
#plot(model, to_file='model.png')

def scheduler(epoch):
    if (epoch + 1) % 10 == 0:
        lr_val = model.optimizer.lr.get_value()
        model.optimizer.lr.set_value(lr_val*0.5)
    return float(model.optimizer.lr.get_value())

sgd = SGD(lr=0.005, clipnorm=40.)
#adam = Adam(clipnorm = 40.)
            
print('Compiling model...')
        
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[categorical_accuracy])
print('Compilation done...')


# Train / Validation Split
choices = np.random.choice(len(train_facts), len(train_facts), replace=False)
train_facts = np.array(train_facts)
valid_facts = train_facts[choices[-len(train_facts)*0.1:]]
train_facts = train_facts[choices[:len(train_facts)*0.9]]

inputs_valid, queries_valid, answers_valid = vectorize_facts(valid_facts, word_idx, 
                                                             story_maxlen, query_maxlen, facts_maxlen,
                                                             enable_time=enable_time)

BATCH_SIZE = 32

show_batch_interval = 1000

linear_regime = True

EPOCHS = 50
N_BATCHS = len(train_facts) // BATCH_SIZE
EARLY_STOP_MAX = 4

save_hist = []
save_hist.append(0.)

for k in xrange(EPOCHS):
    for b,batch in enumerate(grouper(train_facts, BATCH_SIZE, fillvalue=train_facts[-1])):
        inputs_train, queries_train, answers_train = vectorize_facts(batch, word_idx, 
                                                                     story_maxlen, query_maxlen, facts_maxlen,
                                                                     enable_time=enable_time)
        loss = model.train_on_batch([inputs_train, queries_train], answers_train)
        if b % show_batch_interval == 0:
            print('Epoch: {0}, Batch: {1}, loss: {2} - acc: {3}'.format(k, 
                                                                        b, float(loss[0]), float(loss[1])))

    losses = model.evaluate([inputs_valid, queries_valid], answers_valid, batch_size=BATCH_SIZE, 
                            verbose=0)
    print('Epoch {0}, valid loss / valid accuracy: {1} / {2}'.
           format(k, losses[0], losses[1]))
    
       
    #Saving model
    if max(save_hist) < losses[1]:
        model.save_weights('models/weights_memn2n.hdf5', overwrite=True)
    save_hist.append(losses[1])
    
    if max(save_hist) > losses[1] and linear_regime:
        print ('Changing from linear regime ...')
        o1,layers1 = hop_layer(fact_input, question_encoder, adjacent=question_layer, input_layer=True)
        o2,layers2 = hop_layer(fact_input, o1, adjacent=layers1[1])
        o3,layers3 = hop_layer(fact_input, o2, adjacent=layers2[1])

        response = Dense(vocab_size, init='normal', activation=mod_softmax, bias=False)(o3)

        model = Model(input=[fact_input, question_input], output=[response])
        model.load_weights('models/weights_memn2n.hdf5')
        sgd = SGD(lr=0.01, clipnorm=40.)
    
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[categorical_accuracy])
        linear_regime = False
        k = 0
        print ('Done.')

    if max(save_hist) > losses[1]:
        early_stop += 1
    else:
        early_stop = 0
    # Reduce attention loss weights at each early stop call (only two times)
    #if early_stop >= EARLY_STOP_MAX:
    #    break
    scheduler(k)


print('Total Model Accuracy: ')
loss, acc = model.evaluate([inputs_test, queries_test], answers_test, verbose=2)
print('Loss: {0}, Acc: {1}'.format(loss, acc))
print('Per-Task Accuracy: ')
passed = 0
for k, challenge in enumerate(challenge_files):
    test_fact = test_facts_split[k]
    print(challenge)
    inputs_test, queries_test, answers_test = vectorize_facts(test_fact, word_idx, story_maxlen, query_maxlen, facts_maxlen,
                                                         enable_time=enable_time)
    loss, acc = model.evaluate([inputs_test, queries_test], answers_test,  verbose=2)
    print('\n Loss: {0}, Acc: {1}, Pass: {2} '.format(loss, acc, acc >= 0.95))
    passed += acc >= 0.95
print ('Passed: {0}'.format(passed))