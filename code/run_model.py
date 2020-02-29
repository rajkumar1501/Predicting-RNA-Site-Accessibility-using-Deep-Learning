# -*- coding: utf-8 -*-

import pickle
import keras
import tensorflow as tf
from tensorflow.python.keras.models import Model, Input,Sequential
from tensorflow.python.keras.layers import Masking, GRU,Embedding, Dense, TimeDistributed,Bidirectional,concatenate, Dropout, LSTM
from tensorflow.python.keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras import backend  as K
from keras.layers.core import Flatten, Reshape


AA_list = ['U','C','A','G']
aa_index = {char:AA_list.index(char) for char in AA_list}

# The first indices are reserved
aa_index = {k:(v+2) for k,v in aa_index.items()} 
aa_index["<PAD>"] = 0
aa_index["<START>"] = 1


reverse_aa_index = dict([(value, key) for (key, value) in aa_index.items()])
def aa_review(text):
    return [aa_index.get(i,'?') for i in text]


def decode_aa_review(t):
    return ' '.join([reverse_aa_index.get(i,'?') for i in text])



maxlen = 4381





Input = Input(shape=(maxlen,))
x1 = Embedding(input_dim=6, output_dim=64, input_length=maxlen)(Input)
conv =  Convolution1D(filters=64,kernel_size=100,
                             padding='same',activation='relu',name='conv1')(x1)


x = Bidirectional(GRU(units=256, return_sequences=True, recurrent_dropout=0.5))(conv)
conca_output = concatenate([conv, x,x1])
x_1 = Dropout(0.5)(conca_output)
y = TimeDistributed(Dense(3, activation="softmax"))(x_1)
model = Model(Input, y)



def q3_acc(y_true, y_pred):
    y = tf.argmax(y_true, axis=-1)
    y_ = tf.argmax(y_pred, axis=-1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", q3_acc])
model.load_weights('best_model_3.h5')




    

import numpy as np  
def onehot_to_seq(oh_seq, index,length):
    s = ''
    oh_seq = oh_seq.reshape(4381,3)
    prob = oh_seq[:length,2]
    #class_p = oh_seq[:length,1:]
    for i in range(length):
        i = np.argmax(oh_seq[i])
        
        if i != 0:
            s += index[i]
        else:
            break
    return s,prob


def seq_predict(seq_1):

    
    revsere_decoder_index = {1:'b',2:'f'}#{value:key for key,value in {'b':1,'f':2}}
    revsere_encoder_index = dict([(value, key) for (key, value) in aa_index.items()])  
    seq_2 = aa_review(seq_1)
    seq_2=seq_2+[0 for i in range(4381-len(seq_2))]
    seq_2=np.array([seq_2])
    y_train_pred = model.predict(seq_2)
    return onehot_to_seq(y_train_pred,revsere_decoder_index,len(seq_1))
    
    
    
seq='AGGAAAGUCCCGCCUCCAGAUCAAGGGAAGUCCCGCGAGGGACAAGGGUAGUACCCUUGGCAACUGCACAGAAAACUUACCCCUAAAUAUUCAAUGAGGAUUUGAUUCGACUCUUACCUUGGCGACAAGGUAAGAUAGAUGAAGAGAAUAUUUAGGGGUUGAAACGCAGUCCUUCCCGGAGCAAGUAGGGGGGUCAAUGAGAAUGAUCUGAAGACCUCCCUUGACGCAUAGUCGAAUCCCCCAAAUACAGAAGCGGGCUU'
y_pred,proba_free = seq_predict(seq)

proba_free = proba_free.tolist()

with open('../results/result.txt','w+') as f:
    f.write(seq+'\n'+'\n'+'\n')
    f.write(y_pred.upper()+'\n'+'\n'+'\n')
    for i in proba_free:
        f.write('%0.2f ' % i)