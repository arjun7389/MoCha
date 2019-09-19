# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:02:43 2019

@author: Arjun
"""

from keras import callbacks, optimizers
import tqdm
from keras.models import Model
from Dense_Variational import *
import tensorflow_probability as tfp
import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.layers import dot
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
import tensorflow as tf
import urllib.request
import collections
import os
import zipfile

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def maybe_download(filename, url, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


url = 'http://mattmahoney.net/dc/'
filename = maybe_download('D:/L3S/Mocha/Word2Vec/text8.zip', url, 31344016)
vocabulary = read_data(filename)
vocabulary_size=10000
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                                vocabulary_size)
window_size = 2
vector_dim = 100
epochs = 200000
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

vocab_size=10000
sampling_table = sequence.make_sampling_table(vocab_size)
couples, labels = skipgrams(data, vocab_size, window_size=window_size, sampling_table=sampling_table)
word_target, word_context = zip(*couples)
del vocabulary
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")

noise = 1.0
train_size=vocabulary_size
Batch_size = 64
num_batches = train_size / Batch_size
kl_loss_weight = 1.0 / num_batches


input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)
similarity = dot([target, context], axes=0, normalize=True)
dot_product = dot([target, context], axes=1,normalize=False)
dot_product = Reshape((1,))(dot_product)
output = DenseVariational(1, kl_loss_weight=kl_loss_weight,activation='sigmoid')(dot_product)
model = Model(input=[input_target, input_context], output=output)
'''
x_in = Input(shape=(100,))
x = DenseVariational(20, kl_loss_weight=kl_loss_weight, activation='relu')(x_in)
x = DenseVariational(20, kl_loss_weight=kl_loss_weight, activation='relu')(x)
x = DenseVariational(1, kl_loss_weight=kl_loss_weight)(x)

model = Model(x_in, x)
'''
def neg_log_likelihood(y_obs, y_pred, sigma=noise):
    dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
    return K.sum(-dist.log_prob(y_obs))



model.compile(loss=neg_log_likelihood, optimizer=optimizers.Adam(lr=0.03), metrics=['mse'])
validation_model = Model(input=[input_target, input_context], output=similarity)

model.summary()

class SimilarityCallback:
    def run_sim(self):
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            #print(sim)
            nearest = (-sim).argsort()[1:top_k + 1]
            print(nearest)
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        for i in range(vocab_size):
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim
sim_cb = SimilarityCallback()


arr_1 = np.zeros((1,))
arr_2 = np.zeros((1,))
arr_3 = np.zeros((1,))


for cnt in tqdm.tqdm(range(10000)):
    '''
    loss=model.fit([word_target,word_context],labels,batch_size=Batch_size,epochs=1000)
    print("Iteration {}, loss={}".format(cnt, loss))
    sim_cb.run_sim()
    '''
    idx = np.random.randint(0, len(labels)-1)
    arr_1[0,] = word_target[idx]
    arr_2[0,] = word_context[idx]
    arr_3[0,] = labels[idx]
    loss = model.train_on_batch([arr_1, arr_2], arr_3)
    #if cnt % 100 == 0:
    print("Iteration {}, loss={}".format(cnt, loss))
    if cnt % 100 == 0:
        sim_cb.run_sim()
    






