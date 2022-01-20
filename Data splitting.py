# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:35:16 2021

Author: Kline Oware
Script overview:
this script is used to split prepared datasets into;
1. Training
2. Testing 
3. Validation
for the given model generated

"""
#Import required libraries and classes
import numpy as np
import random
from tkinter import filedialog

###### OPENS DIALOBOX TO SELECT FILE#######
w=filedialog.askopenfilename()#read file
data=np.loadtxt(w,delimiter=',')
np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)
#OBTAINS THE DATA SHAPE IN ROWS x COLUMNS
g=data.shape[0]
r=int(0.7*g)
#USES 30% OF THE OVERALL DATA AS TESTING
test=data[r:g,:]
# USES 70% OF THE OVERALL DATA AS TRAINING
train=data[0:r,:]
k=int(0.7*r)
# OBTAINS 30% OF THE TRAINING DATA AS VALIDATION
val=train[k:r,:]
Train=train[0:k,:]
######### THE FILES ARE SAVED AS .csv #########
np.savetxt('Test.csv',test,delimiter=',')
np.savetxt('Train.csv',Train,delimiter=',')
np.savetxt('Validate.csv',val,delimiter=',')
print(" finished!!")
