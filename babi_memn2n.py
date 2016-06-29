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
from keras.layers.core import Dense, Merge, Dropout, RepeatVector, Lambda, Permute, Activation
from keras.layers import recurrent, Input, merge
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, Callback
import keras.backend as K

from keras import initializations, regularizers, constraints

from functools import reduce
import tarfile
import numpy as np
np.random.seed(1337)  # for reproducibility

import re
import pdb

class SequenceEmbedding(Embedding):
    def __init__(self, input_dim, output_dim, position_encoding=False, **kwargs):
        self.position_encoding = position_encoding
        super(SequenceEmbedding, self).__init__(input_dim, output_dim, **kwargs)
    
    
    def call(self, x, mask=None):
        out = super(SequenceEmbedding, self).call(x, mask=mask)
        return K.sum(out, axis=2)

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
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append([sent])
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data

def vectorize_facts(data, word_idx, story_maxlen, query_maxlen, fact_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = np.zeros((len(story), fact_maxlen),dtype='int32')
        for k,facts in enumerate(story):
            x[k][-len(facts):] = np.array([word_idx[w] for w in facts])[:fact_maxlen]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)

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
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
}
challenge_type = 'single_supporting_fact_10k'
challenge = challenges[challenge_type]

EMBED_HIDDEN_SIZE = 64

print('Extracting stories for the challenge:', challenge_type)
train_facts = get_stories(tar.extractfile(challenge.format('train')))
test_facts = get_stories(tar.extractfile(challenge.format('test')))

train_stories = [(reduce(lambda x,y: x + y, map(list,fact)),q,a) for fact,q,a in train_facts]
test_stories = [(reduce(lambda x,y: x + y, map(list,fact)),q,a) for fact,q,a in test_facts]

vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train_stories + test_stories)))
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1

facts_maxlen = max(map(len, (x for h,_,_ in train_facts + test_facts for x in h)))

story_maxlen = max(map(len, (x for x, _, _ in train_facts + test_facts)))
query_maxlen = max(map(len, (x for _, x, _ in train_facts + test_facts)))


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

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_facts(train_facts, word_idx, story_maxlen, query_maxlen, facts_maxlen)
inputs_test, queries_test, answers_test = vectorize_facts(test_facts, word_idx, story_maxlen, query_maxlen, facts_maxlen)

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

print('Build model...')

def hop_layer(x, u, adjacent=None):
    '''
        Define one hop of the memory network
    '''
    if adjacent == None:
        layer_encoder_m = SequenceEmbedding(input_dim=vocab_size,
                                  output_dim=EMBED_HIDDEN_SIZE,
                                  input_length=story_maxlen)
    else:
        layer_encoder_m = adjacent
        
    input_encoder_m = layer_encoder_m(x)
    input_encoder_m = Dropout(0.3)(input_encoder_m)

    layer_encoder_c = SequenceEmbedding(input_dim=vocab_size,
                                  output_dim=EMBED_HIDDEN_SIZE,
                                  input_length=story_maxlen)
    input_encoder_c = layer_encoder_c(x)
    input_encoder_c = Dropout(0.3)(input_encoder_c)

    #output: (samples, max_len, embedding_size )    
    # Memory
    memory = merge([input_encoder_m, u],
                    mode='dot',
                    dot_axes=[2, 1])
    layer_memory = Lambda(lambda x: K.softmax(x))
    memory = layer_memory(memory)
    # output: (samples, max_len)

    # Output
    output = merge([memory, input_encoder_c],
                  mode = 'dot',
                  dot_axes=[1,1])
    output = merge([output, u], mode='sum')
    # output: (samples, embedding_size)
    layers = [layer_encoder_m, layer_encoder_c, layer_memory]
    return output, layers

# 2 hop memn2n
fact_input = Input(shape=(facts_maxlen, story_maxlen, ), dtype='int32', name='facts_input')
question_input = Input(shape=(query_maxlen, ), dtype='int32', name='query_input')

question_encoder = Embedding(input_dim=vocab_size,
                               output_dim=EMBED_HIDDEN_SIZE,
                               input_length=query_maxlen)(question_input)

question_encoder = Dropout(0.3)(question_encoder)
question_encoder = Lambda(lambda x: K.sum(x, axis=1),
                         output_shape=lambda shape: (shape[0],) + shape[2:])(question_encoder)

o1,layers = hop_layer(fact_input, question_encoder)
o2,layers = hop_layer(fact_input, o1, adjacent=layers[0])
o3,layers = hop_layer(fact_input, o2, adjacent=layers[0])

# Response
response = Dense(vocab_size, init='uniform',activation='softmax')(o3)

model = Model(input=[fact_input, question_input], output=[response])

print('Compiling model...')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print('Compilation done...')

# Note: you could use a Graph model to avoid repeat the input twice
model.fit([inputs_train, queries_train], answers_train,
           batch_size=32,
           nb_epoch=120,
           validation_data=([inputs_test, queries_test], answers_test))