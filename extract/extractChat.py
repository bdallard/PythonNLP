#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
	Script d'extraction des data Chat (bleu HB & vert BNP) 
			
"""

import numpy as np
import pandas as pd 
import xlrd
import csv
import os 
import re
import nltk 
from collections import defaultdict
    
com=[]
q1=[]
q2=[]
lmoy=[]
l=0
fileCompt=0

for root, dirs, files in os.walk(pathCorpus):
	xlsFiles=[ _ for _ in files if _.endswith('.xlsx') ]
	for xlsFile in xlsFiles :
		fileCompt+=1
		print("\n*** fichier n°", fileCompt ," en cours de traitement ***")
		workbook = xlrd.open_workbook(os.path.join(root,xlsFile))
		worksheet = workbook.sheet_by_name(u'Détail')
		for line in range(worksheet.nrows): 
			q1.append(worksheet.cell_value(rowx=line, colx=7)) 
			q2.append(worksheet.cell_value(rowx=line, colx=8))
			com.append(worksheet.cell_value(rowx=line, colx=10))  
			l+=1
		print("\n*** fichier n°", fileCompt," traité avec succés ***")

print("\n *** listes crées de taille : ",l,"*** \n")
print(len(com), len(q1))
supp=0
for i,n in enumerate(com): 
	if com[i]=='\r\n':
		supp+=1
		del com[i]
		q1.pop(i)
		q2.pop(i)
	elif n=="\\r\\n":
		com.pop(i)
		q1.pop(i)
		q2.pop(i)
	elif com[i]==" \r\n ":
		com.pop(i)
		q1.pop(i)
		q2.pop(i)
	elif com[i]==" \\r\\n ":
		com.pop(i)
		q1.pop(i)
		q2.pop(i)
print("\n*** pre-process du corpus avec ", supp, "lignes supprimés du dataset original *** ")

for i,n in enumerate(q1): 
	if q1[i]=='très satisfait':
		q1[i]=5
	elif q1[i]=='satisfait': 
		q1[i]=4
	elif q1[i]=='peu satisfait':
		q1[i]=3
	elif q1[i]=='moyennement satisfait':
		q1[i]=2
	elif q1[i]=='pas du tout satisfait':
		q1[i]=1
	elif q1[i]==' ':
		q1[i]=0
	else: 
		q1[i]= 0
	
for i,n in enumerate(q2): 
	if q2[i]=='très satisfait':
		q2[i]=5
	elif q2[i]=='satisfait': 
		q2[i]=4
	elif q2[i]=='peu satisfait':
		q2[i]=3
	elif q2[i]=='moyennement satisfait':
		q2[i]=2
	elif q2[i]=='pas du tout satisfait':
		q2[i]=1
	elif q2[i]==' ':
		q2[i]=0
	else: 
		q2[i]=0

b1 = 0.0
b2 = 1.70
b3 = b2
b4 = 3.3333
b5 = b4
b6 = 5.0
for i in range(1, len(q1)):
	moy = float( (int(q1[i])+int(q2[i])) /3)
	if moy>=b1 and moy<b2:
		lmoy.append(-1) # où -1 := négatif
	elif moy>=b3 and moy<b4:
		lmoy.append(0)# où 0 := neutre
	elif moy>=b5 and moy<=b6:
		lmoy.append(1) # où 1 := positif
        
com.pop(0)
print(len(com))
print(len(lmoy))

dic = {}
dic['label'] = lmoy
dic['phrase'] = com

col = pd.Series(np.zeros(len(lmoy)))
dicbis = {}
dicbis['l1'] = lmoy
dicbis['l2'] = col
dicbis['l3'] = col
dicbis['phrase'] = com
new = pd.DataFrame.from_dict(dicbis)
newbis = new[new.phrase != '\r\n']
newbisbis = newbis[newbis.phrase != ' ']

newbisbis.is_copy = False
for i in range(0, int(newbisbis.shape[0])): 
	if(int(newbisbis.iloc[i,0])==-1):
		newbisbis.iloc[i,0]=1
		newbisbis.iloc[i,1]=0
		newbisbis.iloc[i,2]=0
	elif(int(newbisbis.iloc[i,0])==0):
		newbisbis.iloc[i,0]=0
		newbisbis.iloc[i,1]=1
		newbisbis.iloc[i,2]=0
	elif(int(newbisbis.iloc[i,0])==1): 
		newbisbis.iloc[i,0]=0
		newbisbis.iloc[i,1]=0
		newbisbis.iloc[i,2]=1		

print("\n*** pre-process du corpus terminé *** ")
cleanDataSet = newbisbis[newbisbis.phrase != ''] 
cleanDataSet.to_csv("NewTestChatFinalBIS.csv")
print("\n*** Enregistrement des verbatim labélisés terminé *** ")
