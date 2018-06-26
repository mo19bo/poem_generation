'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.layers import Embedding

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation,TimeDistributed
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
from gensim.models.keyedvectors import KeyedVectors
from utils.DataUtils import *
import json
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
####################

TASK_NAME='chinese-poem-generation-jueju'

#########################

embedding_dims=200

using_word2vec=True

maxlen=2

hidden_size=512

batch_size=512

nb_epochs=200


#######################
#corpus_path='/data/datasets/chinese-poetry/'
corpus_path='/root/datasets/qiyi_title/v2/big/'
corpus_train='svd.title'



seqlen=4
#word2vec_path = '/data/word2vec/word2vec/thuNews.5000Novel.subtitle.tvNovel.word.word2vec.negative.ns.200.min.count.100.txt'
word2vec_path='/root/model/thuNews.5000Novel.subtitle.tvNovel.word.word2vec.negative.ns.200.min.count.100.txt'
kfold_weights_path='checkpoint/%s-nn_weights.{epoch:02d}-{acc:.2f}-using_word2vec_%s-corpus_%s.hdf5' %(TASK_NAME,using_word2vec,corpus_train)
model_save_path='poem/'
#word2vec_path="/root/model/GoogleNews-vectors-negative300.bin"
#word2vec_path="/data/word2vec/word2vec/GoogleNews-vectors-negative300.bin"


word2index_dict='word2index'
model_save_path='model/'

early_stopping_step=3

def useword2vec(word_index):
    #KeyedVectors.load_word2vec_format("/data/word2vec/word2vec/GoogleNews-vectors-negative300.bin", binary=True)
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_path)
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dims))
    for word, i in word_index.items():
        if word in word_vectors.vocab:
          embedding_vector = word_vectors[word]
        else:
          embedding_vector = None
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector[:embedding_dims]

    return embedding_matrix

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    #start_index = random.randint(0, len(X.shape[1]) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = u'春'
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(100):
            data=[ word2index[i] for i in list(sentence)]+[ 0 for i in range(maxlen-len(list(sentence)))]

            x_pred = np.array([data])
            #print (x_pred.shape)
            preds = model.predict(x_pred, verbose=0)[0]
            #print(preds.shape)
            next_index = sample(preds, diversity)
            next_char = index2word[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        sys.stdout.write(generated)
        print()



# c
print ('load data')
data_X,data_Y,index2word,word2index,classses=loaddata_textgenerate_seq2char(corpus_path,corpus_train,maxlen)

#print (xxx)
num_class=len(word2index)+1
train=[i for i in range(len(data_X))]

if using_word2vec:
  print ('Using word2vec to initialize embedding layer ... ')
  embedding_mat=useword2vec(word2index)

  embedding_layer = Embedding(embedding_mat.shape[0],
                            embedding_mat.shape[1],
                            weights=[embedding_mat],
                            input_length=maxlen)

else:
  embedding_layer=Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen)
  embedding_mat=np.zeros((10,200))
print('Vectorization...')


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(embedding_layer)

model.add(GRU(200, dropout=0.2, return_sequences=True))
model.add(TimeDistributed(Dense(num_class,activation='softmax')))

#optimizer = Aadm(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam')
callbacks = [
        # EarlyStopping(monitor='loss', patience=early_stopping_step, verbose=1),
        # ModelCheckpoint(kfold_weights_path, monitor='loss', save_best_only=True, mode='min', verbose=1),
        LambdaCallback(on_epoch_end=on_epoch_end)
    ]




model.fit_generator(batchgeneratedata2(data_X,data_Y, train,batch_size,1,num_class),
            steps_per_epoch=(len(data_X) // batch_size),
          epochs=nb_epochs,
            verbose=1,
          shuffle=True ,
          callbacks=callbacks
         )
print ('save model.....')
model_json = model.to_json()

with open(model_save_path+TASK_NAME, "w") as json_file:
    json_file.write(model_json)

print ('save word2index,index2word ... ...')

with open(model_save_path+word2index_dict, "w") as word2index_vocab:
    json.dump(word2index,word2index_vocab)

model.save_weights(model_save_path+"model.h5")
print("Saved model to disk")

print("Saved model and word2index to %s"% (model_save_path))