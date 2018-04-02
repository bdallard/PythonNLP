# -*- coding: utf-8 -*-

"""

Model de CNN pour classification de phrase (positif, négatif, neutre)

Sources & Inspirations : 
		- http://adventuresinmachinelearning.com/gensim-word2vec-tutorial/
		- https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-7
		- https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
		- https://github.com/keras-team/keras/blob/master/examples/imdb_cnn_lstm.py

"""

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

MAX_LENGTH = 66
TOP_WORDS = 20000

#test = u"j'ai eu des réponses mais le conseillé n'a rien fait..."
#test = u"Super service ! J'adore vraiment je suis LOVE !! "
test = u"Que de la merde ! vraiment nul "
#test = u"Je ne suis pas très convaincu par ce forfait. Néanmoins, l'application est plutôt bien faite!"
#test = u"Il s'agit d'un film moyen sans prétentions. Il est agréable à regarder le soir, mais je n'irai pas jusqu'à dire qu'il était incroyable."
#test = u"Probablement l'un des meilleurs films de la séries ! Un scénario impeccable, des graphismes époustouflants, un jeu d'acteur inégalité malgré le petit budget du réalisateur. A regarder d'urgence."


print("** Starting to tokenize.")
text = test.lower()
tokens = nltk.word_tokenize(text)
print("** Finished to tokenize.")
print tokens
print("** Starting Word2Vec loading.")
W2V = word2vec.load("FR.vocab")
print("** Word2Vec loading ended.")

vector = np.repeat(0, MAX_LENGTH)
for i in range(len(tokens)):
	if tokens[i] in W2V.wv.vocab:
		indexVal = W2V.wv.vocab[tokens[i]].index
		if indexVal < TOP_WORDS:
			vector[i] = indexVal

#Lecture du fichier JSON et création du modèle 
json_file = open('weights/CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

#Load des poids du modèle 
model.load_weights("weights/CNN_weights.h5")
print("Loaded model from disk")
vector = np.asarray([vector])
elapsedTime = time.time()
predict = model.predict(vector)
elapsedTime = time.time() - elapsedTime
interp = abs(predict[0][2] - predict[0][0])

#Affichage 
print "\nTemps de calcul de la prédiction : ({:5.4f}s) ".format(elapsedTime)
print("\n************************************")
print "Positif \t" + "{:5.2f} %".format(predict[0][2] * 100)
print "Neutre \t\t" + "{:5.2f} %".format(predict[0][1] * 100)
print "Négatif \t" + "{:5.2f} %".format(predict[0][0] * 100)
print "Interp. \t" + "{:5.4f}".format(interp)
print("************************************\n")



