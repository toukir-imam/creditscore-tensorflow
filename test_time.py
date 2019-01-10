#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 17:07:12 2018

@author: toukir
"""
from sklearn import svm
import csv
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
f = open('creditcard.csv')
header = f.readline()
print(header)
#data = np.genfromtxt(fname = f,delimiter =',"')
data = list(csv.reader(f, delimiter=",",))
print(len(data))
np_data = np.asarray(data).astype(float)
X = np_data[:,:-1]
y = np.ravel(np_data[:,-1:]).astype(int)
frauds = np_data[np.nonzero(np_data[:,-1:])[0]]
normal = np_data[np.nonzero(1-np_data[:,-1:])[0]]
fraud_x = frauds[:,0]
fraud_y = frauds[:,-2]
normal_x = normal[:,0]
normal_y = normal[:,-2]
plt.scatter(f_x,f_y)