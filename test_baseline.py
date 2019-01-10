#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 10:00:40 2018

@author: toukir
"""

from sklearn import svm
import csv
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
f = open('creditcard.csv')
header = f.readline()
print(header)
#data = np.genfromtxt(fname = f,delimiter =',"')
data = list(csv.reader(f, delimiter=",",))
print(len(data))
np_data = np.asarray(data).astype(float)
X = np_data[:,1:-1]
y = np.ravel(np_data[:,-1:]).astype(int)
frud = np.count_nonzero(y)
score = ((len(y)-frud)/len(y))*100
#result 99.82