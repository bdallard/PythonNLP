#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

    Classification bayesienne binaire (positif, négatif) avec NLTK
    
    Data : échange entre des clients et un SAV, FRANÇAIS
    
    Inspirations : 
                    - http://www.nltk.org/
                    - https://github.com/lesley2958/natural-language-processing/blob/master/intro_nlp.ipynb
                    
"""

import pandas as pd 
import numpy as np 
import timeit
import time
import csv
import os, os.path 
import codecs
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy


dataset = pd.read_csv("EchantillonFaible.csv").values[:,1:]
pos = []
neg = []

for i in range(0, len(dataset[:,0])):
	if(dataset[i,0]==1):
		neg.append(dataset[i,3])
	elif(dataset[i,2]==1):
		pos.append(dataset[i,3])

#tokenization avec NLTK
def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})
    
#définition des ensembles d'entrainement
pos_train = []
neg_train = []
for sentence in pos :
    pos_train.append([format_sentence(sentence), 'positive'])
for sentence in neg :
    neg_train.append([format_sentence(sentence), 'negative'])
training = pos_train[:int((.8)*len(pos_train))] + neg_train[:int((.8)*len(neg_train))]
test = pos_train[:int((.2)*len(pos_train))] + neg_train[:int((.2)*len(neg_train))]

#Classifieur NLTK
classifieur = NaiveBayesClassifier.train(training)

#accuracy
print("*** Accuracy du classifieur ***\n")
print(accuracy(classifieur, test))

#afficher les mots qui "influence" le plus le corpus
print("*** Map des 'features' : mots qui influence le plus le clissifieur ***")
classifieur.show_most_informative_features()

#test 
print("*** Test du classifieur ***\n")
example1 = "C'est vraiment nul je vais me casser d'ici!!"
example2 = "J'adore ce service !!Merci beaucoup c'est super efficace "

print("la phrase : " ,example1, "est : ",classifieur.classify(format_sentence(example1)))
print("la phrase : " ,example2, "est : ",classifieur.classify(format_sentence(example2)))
