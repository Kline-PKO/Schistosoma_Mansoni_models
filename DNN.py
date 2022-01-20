# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 07:13:20 2021

@author: Kline Oware
"""
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from tkinter import filedialog

import time
import tkinter as Tk

import sklearn as sk
import matplotlib.pyplot as plt
from tkinter import filedialog

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from matplotlib import pyplot


#loading data set
print('\n\n SELECT TRAINING DATASET')
print(time.strftime("%b %d %Y %H:%M", time.localtime()))
time.sleep(2)

m=filedialog.askopenfilename(initialdir = '/Desktop',      
 title = '                         SELECT TRAINING DATASET     ')
Train = loadtxt(m,delimiter=',')

n=filedialog.askopenfilename(initialdir = '/Desktop',      
 title = '                         SELECT VALIDATION DATASET   ')
val = loadtxt(n,delimiter=',')

#Splitting datasets into input and output variables
x=Train[:,0:778]
y=Train[:,777]
x_val=val[:,0:778]
y_val=val[:,777]


#Defining the model
def create_model(x,y):
    predict1 = Sequential()
    predict1.add(Dense(520, input_dim=778,activation='relu'))
    predict1.add(Dense(520, activation='relu'))
    predict1.add(Dense(320, activation='sigmoid'))
    predict1.add(Dense(1, activation='sigmoid'))
    predict1.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    weights = {0:1,1:1}
    custom_early_stopping=EarlyStopping(monitor='accuracy',patience=8,min_delta=0.001,mode='max',restore_best_weights=True)
    predict1.fit(x,y,validation_data=(x_val,y_val),epochs=2,batch_size=100,callbacks=[custom_early_stopping])
    return predict1


#Testing
m=filedialog.askopenfilename(initialdir = '/Desktop',      
 title = '                         SELECT TEST DATASET     ')
TEST = loadtxt(m,delimiter=',')
print('    RUNNING PROGRAM\n\n')
time.sleep(3)
m = TEST[:,0:778]
n= TEST[:,777]

#TP,FP,TN,FN=0,0,0,0
#a,ina=0,0
#ac = 0
#inac = 0

#Fit model
model = create_model(m,n)

predictions = model.predict_classes(m)
print('\n\nSTARTING PREDICTIONS\n\n')
time.sleep(3)


#Predict probabilities for test set
yhat_probs = model.predict(m, verbose=0)
#Predict crisp classes for test set
yhat_classes = model.predict_classes(m, verbose=0)


#Reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]

#Rreduce to 1d array
#yhat_probs = yhat_probs[:, 0]
#yhat_classes = yhat_classes[:, 0]

#Accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(n, yhat_classes)
print('Accuracy: %f' % accuracy)
#print('Accuracy: %f' %ACC)

#Balanced accuracy
balanced_accuracy = balanced_accuracy_score(n, yhat_classes)
print('Balanced accuracy: %f' % balanced_accuracy)

#Precision tp / (tp + fp)
precision = precision_score(n, yhat_classes)
print('Precision: %f' % precision)

#Recall: tp / (tp + fn)
recall = recall_score(n, yhat_classes)
print('Recall: %f' % recall)

#Matthew's correlation coefficient
mat = matthews_corrcoef(n, yhat_classes)
print('MCC: %f' % mat)

#f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(n, yhat_classes)
print('F1 score: %f' % f1)

#Kappa
kappa = cohen_kappa_score(n, yhat_classes)
print('Cohens kappa: %f' % kappa)

#ROC AUC
auc = roc_auc_score(n, yhat_probs)
print('ROC AUC: %f' % auc)

#Confusion matrix
matrix = confusion_matrix(n, yhat_classes)
print(matrix)



