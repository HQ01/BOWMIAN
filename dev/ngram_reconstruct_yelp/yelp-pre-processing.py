#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 12:51:20 2018

@author: luzijie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
#from nltk.corpus import stopwords
import string

yelp = pd.read_csv('yelp.csv')
yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
X = yelp_class['text']

f = open('yelp.txt', 'w')

num = 0

for i in X:
    #print(X[i])
    sents = i.split('\n')
    sents = filter(None, sents)
    for j in sents:
        subsent = nltk.sent_tokenize(j)
        for k in subsent:
            #print(k)
            f.write(str(num) + ' ' + k + '\n')
            num+=1
            
f.close()