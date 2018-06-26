from __future__ import print_function

from collections import *
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Merge
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.models import model_from_json
from sklearn.cross_validation import train_test_split, KFold
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np
import jieba
import random


def loaddata_textgenerate_seq2char(corpus_path, corpus_train, seqlen):
    tokenizer = Tokenizer(filters=u'',
                          lower=True,
                          split=" ",
                          char_level=True)
    DATA= [ '@'+i.strip('\n') + '$' for i in open(corpus_path + corpus_train).readlines()]
    #print (DATA)
    tokenizer.fit_on_texts(DATA)

    data_X=[]
    data_Y=[]
    for poem in DATA:
        n_chars = len(poem)
        for i in range(0, n_chars - seqlen, 1):
            s_out = poem[i + seqlen]
            s_in = poem[i:i + seqlen]
            #print (s_in,s_out)

            data_X.append(s_in)
            data_Y.append(s_out)


    X=np.array(tokenizer.texts_to_sequences(data_X))
    Y=np.array(tokenizer.texts_to_sequences(data_Y))
    #print (len(X))


    index2word = dict(zip(tokenizer.word_index.values(), tokenizer.word_index.keys()))
    word2index = tokenizer.word_index
    print(len(Counter(data_Y)), len(word2index))
    #y = np.zeros((len(X), len(word2index)+1))
    #print (len(X), len(word2index))
    #print (xx)
    #print (Y)
    #print (xxx)p
    #print (len(Y))
    # for i,j in enumerate(Y):
    #     #print(i,j)
    #     y[(i,j)]=1
    print (X[0],Y[0])


    return X,Y,index2word,word2index,Counter(data_Y).keys()

def loaddata_inference(corpus,maxlen,word2index):

    X=[ ]
    for i in open(corpus, 'r').readlines():
        line=[]
        for w in jieba.cut(i.split('\t')[0]):
            if w in word2index:
                line.append(word2index[w])
            else:
                line.append(0)
        print (line," ".join(jieba.cut(i.split('\t')[0])))
        X.append(line)


    X_data=sequence.pad_sequences(X, maxlen=maxlen)
    return X_data


def loaddata_inferencebychar(corpus, maxlen, word2index):
    X = []
    for i in open(corpus, 'r').readlines():
        line = []
        for w in list(i.split('\t')[0]):
            if w in word2index:
                line.append(word2index[w])
            else:
                line.append(0)
        #print(line, " ".join(list(i.split('\t')[0])))
        X.append(line)

    X_data = sequence.pad_sequences(X, maxlen=maxlen)
    return X_data
def loaddataformultipleinput(corpus_path, corpus_train,corpus_test,maxlen,label='single',valiadation=0,multiple_kernel=6):

    deliter='_label_'
    deliter2='_deliter_'
    tokenizer = Tokenizer(filters=u'!"#$%&()*+,-./:;<=>?@[\\]^_`，。{|}~\t\n',
                          lower=True,
                          split=" ",
                          char_level=False)
    if label=='single':
        Y ,X1,X2 = zip(*[ (i.split(deliter)[0],i.split(deliter)[1].split('\t')[0],i.split(deliter)[1].split('\t')[1]) for i in open(corpus_path + corpus_train).readlines()])

        _Y,_X1,_X2 = zip(*[(i.split(deliter)[0],i.split(deliter)[1].split('\t')[0],i.split(deliter)[1].split('\t')[1]) for i in open(corpus_path + corpus_test).readlines()])
    #print (Y)
        lb = preprocessing.LabelBinarizer()
        lb.fit(Y)
        Y_DATA=lb.transform(Y+_Y)
    elif label=='multiple':
        Y, X = zip(*[ [i.split(deliter)[0].split(deliter2),i.split(deliter)[1] ] for i in open(corpus_path + corpus_train).readlines()])
        _Y, _X = zip(*[ [i.split(deliter)[0].split(deliter2),i.split(deliter)[1] ] for i in open(corpus_path + corpus_test).readlines()])

        lb = MultiLabelBinarizer()
        #print (Counter([ j for i in Y for j in i]).most_common(20))
        #print (xxx)
        Y_DATA=lb.fit_transform(Y+_Y)
    else:
        pass
        lb=None
        Y_DATA, Y, X, _Y, _X=['','','','','']


    tokenizer.fit_on_texts(X1+X2)
    Train=[]
    Test=[]
    for i in range(multiple_kernel):
        X_DATA_1=sequence.pad_sequences(tokenizer.texts_to_sequences(X1) , maxlen=maxlen)
        X_DATA_2=sequence.pad_sequences(tokenizer.texts_to_sequences(X2) , maxlen=maxlen)
        Train.append(X_DATA_1)
        Train.append(X_DATA_2)
    for j in range(multiple_kernel):
        X_DATA_1=sequence.pad_sequences(tokenizer.texts_to_sequences(_X1) , maxlen=maxlen)
        X_DATA_2=sequence.pad_sequences(tokenizer.texts_to_sequences(_X2) , maxlen=maxlen)
        Test.append(X_DATA_1)
        Test.append(X_DATA_2)
    index2word = dict(zip(tokenizer.word_index.values(), tokenizer.word_index.keys()))
    word2index = tokenizer.word_index
    nums_show = 10
    #print (index2word)
    for x in np.random.choice(len(Train[0]), nums_show, replace=False):
        # print (x)
        print("sample %s in train:"%x )
        print ( Train[0][x],X1[x],Y_DATA[x])
        print( Train[1][x], X2[x],Y_DATA[x])

    return (Train,Y_DATA[:len(Y)]),(Test,Y_DATA[len(Y):]),(X1,X2) ,(word2index, index2word),(list(lb.classes_))
def loaddata(corpus_path, corpus_train,corpus_test,maxlen,label='single',valiadation=0):

    deliter='_label_'
    deliter2='_deliter_'
    tokenizer = Tokenizer(filters=u'!"#$%&()*+,-./:;<=>?@[\\]^_`，。{|}~\t\n',
                          lower=True,
                          split=" ",
                          char_level=False)
    if label=='single':
        Y ,X = zip(*[ (i.split(deliter)[0],i.split(deliter)[1].replace(' ',' ')) for i in open(corpus_path + corpus_train).readlines()])

        _Y,_X = zip(*[ (i.split(deliter)[0],i.split(deliter)[1].replace(' ',' ')) for i in open(corpus_path + corpus_test).readlines()])
    #print (Y)
        lb = preprocessing.LabelBinarizer()
        lb.fit(Y)
        Y_DATA=lb.transform(Y+_Y)
    elif label=='multiple':
        Y, X = zip(*[ [i.split(deliter)[0].split(deliter2),i.split(deliter)[1] ] for i in open(corpus_path + corpus_train).readlines()])
        _Y, _X = zip(*[ [i.split(deliter)[0].split(deliter2),i.split(deliter)[1] ] for i in open(corpus_path + corpus_test).readlines()])

        lb = MultiLabelBinarizer()
        #print (Counter([ j for i in Y for j in i]).most_common(20))
        #print (xxx)
        Y_DATA=lb.fit_transform(Y+_Y)
    else:
        pass
        lb=None
        Y_DATA, Y, X, _Y, _X=['','','','','']


    tokenizer.fit_on_texts(X)

    X_DATA=sequence.pad_sequences(tokenizer.texts_to_sequences(X+_X) , maxlen=maxlen)

    index2word = dict(zip(tokenizer.word_index.values(), tokenizer.word_index.keys()))
    word2index = tokenizer.word_index
    nums_show = 10
    #print (index2word)
    for x in np.random.choice(len(X_DATA), nums_show, replace=False):
        # print (x)
        print("sample %s in train:"%x )
        print ( X_DATA[x],Y_DATA[x])

    return (X_DATA[:len(X)],Y_DATA[:len(Y)]),(X_DATA[len(X):],Y_DATA[len(Y):]) ,(word2index, index2word),(list(lb.classes_))




def batchgeneratedata(data_X,data_Y,train,batch,multiple):
    len_data_x=len(data_X)


    while True:
        random.shuffle(train)
        for i in range(0,len_data_x,batch):

            tr = train[i:i+batch]

            yield [data_X[tr]]*multiple,data_Y[tr]


def batchgeneratedata2(data_X, data_Y,train, batch, multiple,num_class):
    len_data_x = len(data_X)

    while True:
        random.shuffle(train)
        for i in range(0, len_data_x, batch):
            tr = train[i:i + batch]

            yield data_X[tr], to_categorical(data_Y[tr],num_class)