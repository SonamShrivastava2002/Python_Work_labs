# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 11:57:23 2022

@author: sonam
"""

#importing libraries

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

#reading csv files

df_train=pd.read_csv('D:\IC272 DS3\Lab4\SteelPlateFaults-train.csv')
df_test=pd.read_csv('D:\IC272 DS3\Lab4\SteelPlateFaults-test.csv')

#dropping the unnecassary columns

df_train=df_train.drop(df_train.columns[0],axis=1)
df_test=df_test.drop(df_test.columns[0],axis=1)
df_train.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], inplace=True)
df_test.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], inplace=True)

#excluding the class from our test and train data
X_train=df_train[df_train.columns[:-1]]
Y_train=df_train[df_train.columns[-1]]
X_test=df_test[df_test.columns[:-1]]
Y_test=df_test[df_test.columns[-1]]

#splitting the data of both the classes 

df_train_class0 = df_train[df_train['Class']==0]
df_train_class1 = df_train[df_train['Class']==1]
df_test_class0 = df_test[df_test['Class']==0]
df_test_class1 = df_test[df_test['Class']==1]

#counting the shape of data

class0_count=df_train_class0.shape[0]
class1_count=df_train_class1.shape[0]

#excluding  the class from our separate classes data 

X_train_0=df_train_class0[df_train_class0.columns[:-1]]
Y_train_0=df_train_class0[df_train_class0.columns[-1]]
X_train_1=df_train_class1[df_train_class1.columns[:-1]]
Y_train_1=df_train_class1[df_train_class1.columns[:-1]]

#print(x_train_0.columns)

# defining likelihood function

def likelihood(x_vector,mean_vector,cov_matrix): 
    matrix1= np.dot((x_vector-mean_vector).T,np.linalg.inv(cov_matrix))
    power= -0.5*np.dot(matrix1,(x_vector-mean_vector))
    exp_part=np.exp(power)
    # exp_part=bigfloat.exp(pow,bigfloat.precision(100))
    return (exp_part/((2*np.pi)**11.5 * (abs(np.linalg.det(cov_matrix)))**.5))

#defining mean and covariance for both the classes data

mean_0=np.array(X_train_0.mean())
mean_1=np.array(Y_train_1.mean())
cov_0=np.cov((X_train_0).T)
cov_1=np.cov((X_train_1).T)

# To store predictions, we define empty list

bayes_prediction=[] 
for row in X_test.itertuples(index=False):
        l0=likelihood(np.array(row),mean_0,cov_0)
        l1=likelihood(np.array(row),mean_1,cov_1)
        if l0>l1:
            bayes_prediction.append(0)
        else:
            bayes_prediction.append(1)
            
print()
print()
print("\nBayes Classification Predictions:")
confMatrix=confusion_matrix(Y_test, bayes_prediction)
print(confMatrix)
accuracy=accuracy_score(Y_test, bayes_prediction)
print("Accuracy",accuracy*100)
