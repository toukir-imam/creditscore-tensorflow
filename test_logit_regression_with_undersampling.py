#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 10:47:37 2018

@author: toukir

This file does logit regression on the dataset with undersampling
"""

from sklearn import svm
from sklearn import linear_model
import csv
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#read data
f = open('creditcard.csv')
header = f.readline()
print(header)
data = list(csv.reader(f, delimiter=",",))

np_data = np.asarray(data).astype(float)
X = np_data[:,1:-1]
y = np.ravel(np_data[:,-1:]).astype(int)
X = preprocessing.scale(X)

kf = KFold(n_splits=5,shuffle=True)
kf.get_n_splits(X)
accuracy_score =[] 

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    
    #undersample
    fraud_idx = np.nonzero(y_train)[0]
    normal_idx = np.nonzero(1-y_train)[0]

    fraud_X = X_train[fraud_idx]
    fraud_y = y_train[fraud_idx]
    normal_X = X_train[normal_idx]
    normal_y = y_train[normal_idx]

    rand_choice = np.random.randint(normal_X.shape[0], size=40000)
    normal_X = normal_X[rand_choice,:]
    normal_y = normal_y[rand_choice]
    X_train = np.vstack((normal_X,fraud_X))
    y_train = np.hstack((normal_y,fraud_y))
    
    #build model
    clf = linear_model.SGDClassifier(loss='log')


    #train model
    clf.fit(X_train,y_train)
    
    print('training done')
    
    #predict
    accuracy_score.append(clf.score(X_test,y_test))

    
    #score = clf.score(X_test,y_test)
    #print ("Accuracy : " + str(np.mean(scores['test_acc'])))
    #print ("Precision : " + str(np.mean(scores['test_precision'])))
    #print ("Recall : " + str(np.mean(scores['test_recall'])))
    #print ("F1 : " + str(np.mean(scores['test_f1'])))
#print accuracy
print(accuracy_score)
print(np.mean(accuracy_score))
    
