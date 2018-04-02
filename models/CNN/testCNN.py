#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""

Model de CNN pour classification de phrase (positif, négatif, neutre)

Sources & Inspirations : 
		- http://adventuresinmachinelearning.com/gensim-word2vec-tutorial/
		- https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-7
		- https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
		- https://github.com/keras-team/keras/blob/master/examples/imdb_cnn_lstm.py

"""


import pandas as pd 
import xlrd
import csv
import os
import sys
import gensim
import math
import numpy as np
import sklearn 
import gensim, logging
from gensim.models import word2vec
from keras.datasets import imdb
from keras.layers import LSTM, Convolution1D, Flatten, Dropout, Dense
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.models import Sequential
import nltk
import word2vec
import scipy
import sklearn.linear_model 
from sklearn.cross_validation import train_test_split

sizeMax = 66
ftest = "/test2.csv"
TOP_WORDS = 20000
LOAD_VOCAB = True 
miniBatch_size = 64

f = pd.read_csv("testEchantillonnage1.csv").values[:,1:]
print("\n*** La taille du dataSet est de : ", len(f[:,0]) ," lignes ***")
token = []

for i in range(0, len(f[:,3])): 
	st = str(f[i,3]).lower().decode('utf8')
	token.append([ nltk.word_tokenize(st) ])
print("*** finished to tokenize ***")


print("*** Start Word2Vec with ",len(token), " elements *** ")

if LOAD_VOCAB:
	print("** Starting Word2Vec loading.")
	w2v = word2vec.load('weights/FR.vocab')
	print("** Word2Vec loading ended.")
else:
	print("** Starting Word2Vec saving.")
	w2v = word2vec.save(token, 'weights/FR.vocab')
	print("** Word2Vec saving ended.")

w2v = word2vec.save(token, 'FR.vocab')
print("\n*** Word2Vec processed ***\n")

keys = w2v.wv.vocab.keys()
vocabIndex = {}
for i in range(len(keys)):
	w = keys[i]
	vocabIndex[w] = w2v.wv.vocab[w].index
	if i % 500 == 0:
		print(str((i * 100) / len(keys)) + " % done.")

#Définition des ensembles d'entraienements X_train et Y_train 
X_train = [ np.repeat(0, sizeMax) ]
for i in range(len(token)):
	row = np.repeat(0, sizeMax)
	words = token[i][0]
	for j in range(min(sizeMax, len(words))):
		if words[j] in w2v.wv.vocab:
			indexVal = vocabIndex[words[j]]
			if indexVal < TOP_WORDS:
				row[j] = indexVal
	X_train.append(row)
	if i % 1000 == 0:
		print(str((i * 100) / len(token)) + " % done.")
		
X_train = np.asarray(X_train[1:])
print("** French dataset loaded.")
Y_train = f[:,0:3]
Y_test = f[0:716,0:3]
X_test = X_train[0:716,:]
print("Évaluation shape des vecteurs :  \n")
print(X_train.shape, Y_train.shape, X_test, Y_test)


model = Sequential()
model.add(Embedding(TOP_WORDS, miniBatch_size, input_length=sizeMax))
model.add(Convolution1D(64, 3, padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, epochs=10, batch_size=miniBatch_size)

scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model_json = model.to_json()
with open("weights/CNN_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("weights/CNN_weights.h5")

print("*** Saved model to disk ***")

