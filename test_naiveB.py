#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 10:57:24 2018

@author: toukir
"""

from sklearn import svm
from sklearn import linear_model
import csv
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
f = open('creditcard.csv')
header = f.readline()
print(header)
#data = np.genfromtxt(fname = f,delimiter =',"')
data = list(csv.reader(f, delimiter=",",))
print(len(data))
np_data = np.asarray(data).astype(float)
X = np_data[:,1:-1]
y = np.ravel(np_data[:,-1:]).astype(int)
#X = preprocessing.scale(X)
#clf = linear_model.SGDClassifier(loss='log')
#scores = cross_val_score(clf, X, Y, cv=5)
gnb = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60, random_state=0)
#clf.fit(X_train,y_train)
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print('training done')
score = (len(y_test)-(y_test !=y_pred).sum())/len(y_test)
print(score)