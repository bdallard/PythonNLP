# -*- coding: utf-8 -*-
import sys
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import model_from_json
import numpy as np
import nltk
import pandas as pd
import word2vec
import gensim

#si le vocabulaire est bien généré par le word2vec
LOAD_VOCAB = True
#nombre de mot maximum d'une phrase 
MAX_LENGTH = 15
#nombre de mots maximum 
TOP_WORDS = 20000
#lecture du dataset
dataset = pd.read_csv("trainBNP.csv").values[:,1:]
print("\n*** La taille du dataSet est de : ", len(dataset[:,0]) ," lignes ***")
#début tokenization 
tokens = []
for i in range(len(dataset[:,0])):
	text = str(dataset[i,0]).lower().decode("utf8")
	if i % 2000 == 0:
		print(str((i * 100) / len(dataset[:,0])) + " % done.")
	tokens.append([ nltk.word_tokenize(text) ])
print("** Finish ! **")

if LOAD_VOCAB:
	W2V = word2vec.load("train_result/FR.vocab")
else:
	#on génère le corpus
	W2V = word2vec.save(tokens, "train_result/FR.vocab")

#définition d'un indice pour remplire un dictionnaire classique 
keys = W2V.wv.vocab.keys()
vocabIndex = {}
for i in range(len(keys)):
	#les objets word2vec vocab sont compliqué à gérer 
	w = keys[i]
	vocabIndex[w] = W2V.wv.vocab[w].index
	if i % 500 == 0:
		print(str((i * 100) / len(keys)) + " % done.")
#définition de l'ensemble d'entrainement 
X_train = [ np.repeat(0, MAX_LENGTH) ]
for i in range(len(tokens)):
	row = np.repeat(0, MAX_LENGTH)
	words = tokens[i][0]
	#remplissage avec les mots
	for j in range(min(MAX_LENGTH, len(words))):
		#si le mot est dans le corpus 
		if words[j] in W2V.wv.vocab:
			indexVal = vocabIndex[words[j]]
			if indexVal < TOP_WORDS:
				row[j] = indexVal
	X_train.append(row)
	#print de l'avancement
	if i % 1000 == 0:
		print(str((i * 100) / len(tokens)) + " % done.")
		
X_train = np.asarray(X_train[1:])
Y_train = dataset[:,0:3]
#ensemble de test 
X_test = X_train[0:5530,:]
Y_test = Y_train[0:5530,:]

#create the model
embedding_vecor_length = 64
model = Sequential()
model.add(Embedding(TOP_WORDS, embedding_vecor_length, input_length=MAX_LENGTH))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, epochs=10, batch_size=64)
#évaluation du model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#serialization du model en anglais JSON
model_json = model.to_json()
with open("train_result/FR_LSTM_model.json", "w") as json_file:
    json_file.write(model_json)
#enregistrement des poids
model.save_weights("train_result/FR_LSTM_weights.h5")
print("Saved model to disk")
