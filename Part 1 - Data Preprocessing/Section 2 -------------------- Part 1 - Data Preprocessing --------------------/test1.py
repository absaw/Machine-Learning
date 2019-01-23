# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 20:03:33 2018

@author: admin
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as mp
import pandas as pd

#Creating a dataset variable and importing the dataset.csv file with it
dataset = pd.read_csv('Data.csv')

#Creating separate matrices for x - indep, y- dep 
x = dataset.iloc[ : , :3].values
y = dataset.iloc[ : , 3].values

#ELIMINATING THE MISSING VALUES

from sklearn.preprocessing import Imputer
#creating function variable using Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
#attaching the variable imputer to our matrix x
imputer = imputer.fit(x[:,1:3])
#now we apply our imputer variable on matrix to fill in the
#missing values will be filled with the strategy we picked
#fit() used to apply changes on a temp var in memory
#transform() used to commit the changes to the said variable
#fit_transform() for doing both together
x[:,1:3]=imputer.transform(x[:,1:3])

#CATEGORIZING COLUMNS
""" The LabelEncoder
-we want to encode different entries of a column so they can be 
uniquely identified while writing equations
we again import sklearn.preprocessing Label Encoder
-LE just assigns a number to each different entry serially-0,1,2..
-but this gives a weightage to the entries which is not needed.
-Thus we will use dummy encoding to give equal weightage to every
entry 
"""
"""  The OneHotEncoder
-Here the one hot encoder is used to implement dummy encoding on an independent 
column.
-What it does is make the column entry 1 if an entry exists belonging to that 
column, while all others are 0. 
-Thus it adds the n no. of columns = no. of possible values of the column
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder as ohe
labelEncoderX = LabelEncoder()
x[:,0]=labelEncoderX.fit_transform(x[:,0])

onehotencoder_x = ohe(categorical_features=[0])
x=onehotencoder_x.fit_transform(x).toarray()

labelEncodery = LabelEncoder()
y = labelEncodery.fit_transform(y)

#Splitting dataset to training and test set
'''
Training Set- from which the model will learn from
Test -with which it will compare itself and check itself

'''
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,test_size = 0.2, random_state=0)

#Feature Scaling- it scales the entries so that all columns are comparable to 
#same scale
from sklearn.preprocessing import StandardScaler as ss
sc_x = ss()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

