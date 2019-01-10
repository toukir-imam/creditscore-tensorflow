#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 09:54:07 2018

@author: toukir
This file performs support vector classification on the dataset.

"""
from sklearn import svm
import csv
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#read data
f = open('creditcard.csv')
header = f.readline()
print(header)

data = list(csv.reader(f, delimiter=",",))

np_data = np.asarray(data).astype(float)
X = np_data[:,1:-1]
y = np.ravel(np_data[:,-1:]).astype(int)
X = preprocessing.scale(X)

#build model
clf = svm.SVC(kernel='rbf', C=10)

#do 5 fold cross validation
scores = cross_val_score(clf, X, y, cv=5)

print(np.mean(scores)
