# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:29:59 2022

@author: sonam
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 23:20:20 2022

@author: HP
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv('D:\IC272 DS3\Lab5\\abalone.csv')
best_column=df.corr().iloc[-1,:-1].idxmax()
#print(best_column)

df_train,df_test=train_test_split(df,test_size=0.3,random_state=42)
df_train.to_csv("D:\IC272 DS3\Lab5\\abalone-train.csv",index=False)
df_test.to_csv("D:\IC272 DS3\Lab5\\abalone-test.csv",index=False)
x_train=df_train[df_train.columns[:-1]]
y_train=df_train[df_train.columns[-1]]
x_test=df_test[df_test.columns[:-1]]
y_test=df_test[df_test.columns[-1]]

regr = LinearRegression()
X=np.array(x_train[best_column]).reshape(-1, 1)
Y=np.array(y_train).reshape(-1, 1)
regr.fit(X,Y)
X_test=np.array(x_test[best_column]).reshape(-1,1)
y_pred = regr.predict(np.array(X_test))

print(y_pred)
print(y_test)

import matplotlib.pyplot as plt
plt.scatter(X, Y, color ='b')
plt.plot(X_test, y_pred, color ='k')
plt.show()
# Data scatter of predicted values

plt.plot(df_train[best_column],Y)
plt.xlabel("chosen attribute value i.e. "+best_column)
plt.ylabel("Rings")
plt.show()

#que1 partb
# RMSE error for Training data
from sklearn.metrics import mean_squared_error
import math

print(np.array(x_train[best_column]))
MSE = mean_squared_error(regr.predict(np.array(x_train[best_column]).reshape(-1, 1)),np.array(y_train))
 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)

# RMSE error for testing data
MSE = mean_squared_error(y_pred, y_test)
 
RMSE = math.sqrt(MSE)

print("Root Mean Square Error:\n")
print(RMSE)

plt.scatter(y_test,y_pred)
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.show()

#que-2

regr = LinearRegression()
Y=np.array(y_train).reshape(-1, 1)
regr.fit(x_train,Y)
y_pred = regr.predict(x_test)
print(y_pred)

# find the RMSE
MSE = mean_squared_error(regr.predict(x_train), y_train)
 
RMSE = math.sqrt(MSE)

print("Root Mean Square Error:\n")
print(RMSE)

# find the RMSE
MSE = mean_squared_error(y_pred, y_test)
 
RMSE = math.sqrt(MSE)

print("Root Mean Square Error:\n")
print(RMSE)

plt.scatter(y_test,y_pred)
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.show()

# Que-3 Non Linear Regression Model

from sklearn.preprocessing import PolynomialFeatures
P=[2,3,4,5]
least_error=9999
for p in P:
    poly_features = PolynomialFeatures(p) #p is the degree
    x_train_trans = poly_features.fit_transform(X)
    x_test_trans = poly_features.transform(X_test)

    regressor = LinearRegression()
    regressor.fit(x_train_trans, y_train)

    #Input arguments: x_poly: Polynomial expansion of input
    # variable(s) of training data and y: target values
    y_pred = regressor.predict(x_test_trans)
    MSE = mean_squared_error(regressor.predict(x_train_trans),np.array(y_train))
 
    RMSE = math.sqrt(MSE)
    print("Root Mean Square Error with train data for p=",p)
    print(RMSE)
    if(least_error>RMSE):
        least_error=RMSE
        best_fit_y=y_pred
        best_fit_x=X_test

    MSE = mean_squared_error(regressor.predict(x_test_trans),np.array(y_test))
    
    RMSE = math.sqrt(MSE)
    print("Root Mean Square Error with test for p=",p)    
    print(RMSE)
plt.plot(best_fit_x, best_fit_y, "r-", linewidth=2, label="Predictions")
plt.plot(X, y_train, "b.",label='Training points')
plt.plot(X_test, y_test, "b.",label='Testing points')
plt.xlabel("X-The chosen attribute i.e. "+best_column)
plt.ylabel("y-Rings")
plt.legend()
plt.show()

plt.scatter(y_test,y_pred)
plt.xlabel("Actual Rings")
plt.ylabel("Predicted")
plt.show()

# Que-4

P=[2,3,4,5]
least_error=9999
Bar_graph_x_train=[]
Bar_graph_y_train=[]
Bar_graph_x_test=[]
Bar_graph_y_test=[]
for p in P:
    poly_features = PolynomialFeatures(p) #p is the degree
    x_train_trans = poly_features.fit_transform(x_train)
    x_test_trans = poly_features.transform(x_test)

    regressor = LinearRegression()
    regressor.fit(x_train_trans, y_train)

    #Input arguments: x_poly: Polynomial expansion of input
    # variable(s) of training data and y: target values
    y_pred = regressor.predict(x_test_trans)
    MSE = mean_squared_error(regressor.predict(x_train_trans),np.array(y_train))
 
    RMSE = math.sqrt(MSE)
    print("Root Mean Square Error with train data for p=",p)
    print(RMSE)
    Bar_graph_x_train.append(p)
    Bar_graph_y_train.append(RMSE)
    if(least_error>RMSE):
        least_error=RMSE
        best_fit_y=y_pred
        best_fit_x=X_test

    MSE = mean_squared_error(regressor.predict(x_test_trans),np.array(y_test))
    
    RMSE = math.sqrt(MSE)
    print("Root Mean Square Error with test for p=",p)    
    print(RMSE)
    Bar_graph_x_test.append(p)
    Bar_graph_y_test.append(RMSE)
plt.bar(Bar_graph_x_train,Bar_graph_y_train)
plt.show()
plt.bar(Bar_graph_x_test,Bar_graph_y_test)
plt.show()    

plt.scatter(y_test,best_fit_y)
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.show()
