# -*-coding: utf-8 -*
import os
import sys
import gensim
import math
import numpy as np
import gensim, logging
from gensim.models import word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def save(tokens, filename, size=100):
	vocab = []
	for i in range(len(tokens)):
		for j in range(len(tokens[i])):
			vocab.append(tokens[i][j])
	model = word2vec.Word2Vec(vocab, size=size, min_count=3, window=5, workers=2)
	model.save(filename)
	return model

def load(filename):
	return word2vec.Word2Vec.load(filename)

