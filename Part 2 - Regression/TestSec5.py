# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:07:01 2019

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#we have a categorical column in x that we need to create dummy variables of
from sklearn.preprocessing import LabelEncoder, OneHotEncoder as ohe
labelEncoderX = LabelEncoder()
X[:,3]=labelEncoderX.fit_transform(X[:,3])

onehotencoder_X = ohe(categorical_features=[3])
X=onehotencoder_X.fit_transform(X).toarray()

#Not falling in the dummy variable trap
X=X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
    
#Predicting the output using x-test values

y_pred = regressor.predict(X_test)

"""
Cant plot when your x axis has multiple dimensions !!:)
#Plotting
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.xlabel('Independent variables')
plt.ylabel('Profit')
plt.show()
"""

"""
-Now we are using a new model to perform stepwise regr(backward elimination)
to optim
-In this model we need to add the coefficient bo since it wont be included in 
the model
-ASSUMING SIGNIFICANCE LEVEL S = 0.05
""" 
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[ :,:]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#MAP THE COLUMN IN SUMMARY TO REAL COLUMNS OF VARIABLE X TO FIND WHICH COLUMN TO REMOVE NEXT
#In summary x2 col:2 came out to be highest 0.990 so we remove that
X_opt = X[ :,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#In summary x1 col:1 came out to be highest 0.940 so we remove that
X_opt = X[ :,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#In summary x2 WHICH IS ACTUALLY THE 4TH COLUMN came out to be highest 0.602 so we remove that
X_opt = X[ :,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#In summary x3 col 5 came out to be highest 0.06 so we remove that

X_opt = X[ :,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
