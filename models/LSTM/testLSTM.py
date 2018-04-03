# -*- coding: utf-8 -*-
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import model_from_json
import nltk
import word2vec
import gensim
import numpy as np
import time

MAX_LENGTH = 15
TOP_WORDS = 20000

test = u"Super service ! J'adore vraiment je suis LOVE !! "
#test = u"Le jeu d'acteur est très mauvais, les textes sont nuls, bref, encore un autre film bien pourri."
#test = u"Je ne suis pas très convaincu par ce film. Néanmoins, les graphismes sont plutôt bien faits."
#test = u"Il s'agit d'un film moyen sans prétentions. Il est agréable à regarder le soir, mais je n'irai pas jusqu'à dire qu'il était incroyable."
#test = u"Probablement l'un des meilleurs films de la séries ! Un scénario impeccable, des graphismes époustouflants, un jeu d'acteur inégalité malgré le petit budget du réalisateur. A regarder d'urgence."

print("** Starting to tokenize.")
text = test.lower()
tokens = nltk.word_tokenize(text)
print("** Finish ! ** ")
#print tokens
print("** Starting Word2Vec loading.")
W2V = word2vec.load("train_result/FR.vocab")
print("** Finish ! ** ")

vector = np.repeat(0, MAX_LENGTH)
for i in range(len(tokens)):
	if tokens[i] in W2V.wv.vocab:
		indexVal = W2V.wv.vocab[tokens[i]].index
		if indexVal < TOP_WORDS:
			vector[i] = indexVal

#on va chercher le model en JSON
json_file = open('train_result/FR_LSTM_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
#on va chercher les poids
model.load_weights("train_result/FR_LSTM_weights.h5")
print("** model importé **")
vector = np.asarray([vector])
elapsedTime = time.time()
predict = model.predict(vector)
elapsedTime = time.time() - elapsedTime
interp = (predict[0][2] - predict[0][0])

print "\nPrediction ({:5.4f}s) :".format(elapsedTime)
print "Positif \t" + "{:5.2f} %".format(predict[0][2] * 100)
print "Neutre \t\t" + "{:5.2f} %".format(predict[0][1] * 100)
print "Négatif \t" + "{:5.2f} %".format(predict[0][0] * 100)
print "Interp. \t" + "{:5.4f}".format(interp)
