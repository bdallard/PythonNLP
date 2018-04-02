#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
	Script d'extraction + labelisation (positif, négatif, neutre)
	Données enquetes mail 
'''

import pandas as pd 
import xlrd
import csv
import os, os.path 
import codecs
import re
from bs4 import BeautifulSoup
from pathlib import Path
from xlwt import Workbook

com = []
l1 = []
l2 = []
l3 = []
lmoy = []

#parcours des fichier du répertoire 
nbfichier = 0
for fichier in os.listdir(pathTest):
	ext = os.path.splitext(fichier)
		
	#traitement uniquement des fichier .xls
	if ext[1]==".xml":
		nbfichier+=1
		infile=open(os.path.join(pathTest, fichier), "rb")
		content=infile.read()
		print("fichier ", fichier, " lu")
		soup = BeautifulSoup(content,'xml')
		titles = soup.find_all('title')
		mess = soup.find_all('Response')
		txt = soup.find_all('QID6_TEXT')
		q1 = soup.find_all('QID1')
		q2 = soup.find_all('QID2')
		q3 = soup.find_all('QID3')
		print("fichier ", fichier, " traité")
		
		for i in range(0, len(txt)):
			com.append(txt[i].get_text())
			l1.append(q1[i].get_text())	
			l2.append(q2[i].get_text())
			l3.append(q3[i].get_text())


#traitement des tableaux de labels 
for i,n in enumerate(l1):
	if l1[i]=='Très satisfait':
		l1[i]=5
	elif l1[i]=='Satisfait': 
		l1[i]=4
	elif l1[i]=='Peu satisfait':
		l1[i]=3
	elif l1[i]=='Moyennement satisfait':
		l1[i]=2
	elif l1[i]=='Pas du tout satisfait':
		l1[i]=1

for i,n in enumerate(l2):
	if l2[i]=='Très satisfait':
		l2[i]=5
	elif l2[i]=='Satisfait': 
		l2[i]=4
	elif l2[i]=='Peu satisfait':
		l2[i]=3
	elif l2[i]=='Moyennement satisfait':
		l2[i]=2
	elif l2[i]=='Pas du tout satisfait':
		l2[i]=1

for i,n in enumerate(l3):
	if l3[i]=='Très satisfait':
		l3[i]=5
	elif l3[i]== 'TrÃ¨s satisfait':
		l3[i]=5
	elif l3[i]=='Satisfait': 
		l3[i]=4
	elif l3[i]=='Peu satisfait':
		l3[i]=3
	elif l3[i]=='Moyennement satisfait':
		l3[i]=2
	elif l3[i]=='Pas du tout satisfait':
		l3[i]=1

labels1 = ['phrases', 'label1', 'label2', 'label3', 'labelMoy']
data1 = [com, l1, l2, l3, lmoy]
df1= pd.DataFrame.from_records(data1, labels1)

for i in range(0, len(l1)):
	moy = float( (int(l1[i])+int(l2[i])+int(l3[i])) /3)
	if moy>=0 and moy<1.70:
		lmoy.append('négatif') #négatif
	elif moy>=1.70 and moy<3.33:
		lmoy.append('neutre')#neutre
	elif moy>=3.33 and moy<=5:
		lmoy.append('positif') #positif

labels2 = ['phrases', 'labelMoyenne']
data2 = [com, lmoy]
df2= pd.DataFrame.from_records(data2, labels2)
