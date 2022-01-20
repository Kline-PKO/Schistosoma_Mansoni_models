# -*- coding: utf-8 -*-

#Import required libraries
import os
import numpy as np
from numpy import savetxt
import sklearn as sk
import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tkinter import Tk
from tkinter import filedialog
from time import sleep
import random


t=filedialog.askdirectory()
os.chdir(t)
print('\n\n    LET US START BY IMPORTING THE COMPUTED DESCRIPTORS')
sleep(2)
a=filedialog.askopenfilename(initialdir = '/Desktop',      title = '                                    SELECT A FILE TO START PREPROCESSING ')
print('         STARTING DATA PREPROCESSING')

data = pd.read_csv(a)
print('          EXCEL FILE IMPORTED FOR PREPROCESSING\n\n\n')

#Column Removal
#pct_null=data.isnull().sum()/len(data)
#missing_features=pct_null[pct_null>0.49].index
#data.drop(missing_features,axis=1, inplace=True)
#print('          COLUMNS WITH MORE THAN 50% EMPTY SPACES REMOVED\n\n\n')


#Filling empty spaces with 0
data=data.fillna(0)
print('          EMPTY CELLS SUCCESSFULLY FILLED WITH 0 \n\n\n')

#Dropping name axis
#data=data.drop('name',axis=1)
#print('          NAME COLUMN SUCCESFULLY DROPPED\n\n\n')

#Converting to numerical data
numdata = np.array(data,dtype=np.float64)

#Data Standardization
sca = sk.preprocessing.StandardScaler()
sca.fit(numdata)
numstdata = sca.transform(numdata)
print('          DATA CONVERTED TO NUMERICAL TYPE\n\n\n')


r = filedialog.askopenfilename(title='   SELECT ACTIVE COMPOUNDS')
y = filedialog.askopenfilename(title='   SELECT INACTIVE COMPOUNDS')
act = pd.read_csv(r)
inact = pd.read_csv(y)
m=np.ones((act.shape[0],1))
n=np.zeros((inact.shape[0],1))
q = np.concatenate((m,n),axis=0)

# Deleting the first and second columns since they all had the same values
#num1=np.delete(numstdata,[0,1],1)

#Saving the processed data as a csv file
np.savetxt('Processed_descriptors.csv',numstdata,delimiter=',')

np_data=np.concatenate((numstdata,q),axis=1)		


#Reading numpy files
#num1=np.genfromtxt(directory,delimiter=',')

#deleting rows or columns from the array
#num1=np.delete(num1,index,column)
#num1=np.delete(num1,np.s_[index],column) to delete multiple colums: I DID this cos the dataset was mixed with a lot of empty colums
#RANDOM

#SHUFFLING THE DATASETS WITH THE FOLLOWING COMMANDS (69 & 70)
np.take(np_data,np.random.permutation(np_data.shape[0]),axis=0,out=np_data)

np.savetxt('Prepared_Data.csv',np_data,delimiter=',')
#properly sorting out the data

print('          PREPROCESSING FINISH')
