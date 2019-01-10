#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:03:19 2018

@author: toukir

This file does DNN Classification on the creditcard dataset using the Tensorflow library
"""

import tensorflow as tf
import numpy as np
from random import shuffle
from sklearn import svm
import csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold

#read data
f = open('creditcard.csv')
header = f.readline()
print(header)


data = list(csv.reader(f, delimiter=",",))

np_data = np.asarray(data).astype(float)
X = np_data[:,1:-1]
y = np.ravel(np_data[:,-1:]).astype(int)
X = preprocessing.scale(X)


# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[29])]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=[20,50 ,20],
                                        n_classes=2,
                                        )


#do k fold
kf = KFold(n_splits=5,shuffle=True)
kf.get_n_splits(X)
accuracy_score =[] 

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    fraud_idx = np.nonzero(y_train)[0]
    normal_idx = np.nonzero(1-y_train)[0]

    #Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_train},
        y=y_train,
        num_epochs=None,
        shuffle=True)
    print('started training')
    classifier.train(input_fn=train_input_fn, steps=2000)

    #Define test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)

# Evaluate accuracy.
    accuracy_score.append(classifier.evaluate(input_fn=test_input_fn)["accuracy"])

print (accuracy_score)
print("\nTest Accuracy: {0:f}\n".format(np.mean(accuracy_score)))








