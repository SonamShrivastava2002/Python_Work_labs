# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 20:30:02 2022

@author: sonam
"""

#importing libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score


df= pd.read_csv('D:\IC272 DS3\Lab4\SteelPlateFaults-2class.csv')
df_train,df_test=train_test_split(df,test_size=0.3,random_state=42)
df_train.reset_index(inplace=True)
df_test.reset_index(inplace=True)

#excluding the class from our test and train data

x_train=df_train[df_train.columns[:-1]]
y_train=df_train[df_train.columns[-1]]
x_test=df_test[df_test.columns[:-1]]
y_test=df_test[df_test.columns[-1]]

#For KNN classifier data

accuracy1=0
for i in range(1,7,2):
    classifier = KNeighborsClassifier(n_neighbors = i)
    classifier.fit(x_train, y_train)
    y_predict=classifier.predict(x_test)
    cm = confusion_matrix(y_test, y_predict)
    print("Confusion Matrix for Neighbour: ",i)
    print(cm)
    print ("Accuracy : ", accuracy_score(y_test, y_predict))
    accuracy1=max(accuracy_score(y_test, y_predict),accuracy1)
    print()
  
#for KNN normalised data  

df_train=pd.read_csv('D:\IC272 DS3\labcsvfiles\SteelPlateFaults-train.csv')
df_test=pd.read_csv('D:\IC272 DS3\labcsvfiles\SteelPlateFaults-test.csv')
for i in df_train.columns[:-1]:
    df_train[i] = (df_train[i]-df_train[i].min())/(df_train[i].max()-df_train[i].min())
    
for i in df_test.columns[:-1]:
    df_test[i] = (df_test[i]-df_test[i].min())/(df_test[i].max()-df_test[i].min())

#excluding the class from our test and train data

x_train=df_train[df_train.columns[:-1]]
y_train=df_train[df_train.columns[-1]]
x_test=df_test[df_test.columns[:-1]]
y_test=df_test[df_test.columns[-1]]


k_maxAccuracy=0;
maxAccuracy=0;
for i in range(1,7,2):
    classifier = KNeighborsClassifier(n_neighbors = i)
    classifier.fit(x_train, y_train)
    y_predict=classifier.predict(x_test)
    cm = confusion_matrix(y_test, y_predict)
    print("Confusion Matrix for Neighbour: ",i)
    print(cm)
    accuracy=accuracy_score(y_test, y_predict)
    print ("Accuracy : ",accuracy)
    if accuracy>maxAccuracy:
        maxAccuracy=accuracy
        k_maxAccuracy=i
    print()
    
print("Max Accuracy:",maxAccuracy,"for k=",k_maxAccuracy)
accuracy2=maxAccuracy

#for bayes classifier using unimodal gaussian distribution

#dropping the unnecassary columns

df_train=df_train.drop(df_train.columns[0],axis=1)
df_test=df_test.drop(df_test.columns[0],axis=1)
df_train.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], inplace=True)
df_test.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], inplace=True)

#excluding the class from our test and train data

x_train=df_train[df_train.columns[:-1]]
y_train=df_train[df_train.columns[-1]]
x_test=df_test[df_test.columns[:-1]]
y_test=df_test[df_test.columns[-1]]

#splitting the data of both the classes 

df_class0_train = df_train[df_train['Class']==0]
df_class1_train = df_train[df_train['Class']==1]
df_class0_test = df_test[df_test['Class']==0]
df_class1_test = df_test[df_test['Class']==1]

#counting the shape of data

class0_count=df_class0_train.shape[0]
class1_count=df_class1_train.shape[0]

#excluding  the class from our separate classes data 

x_train_0=df_class0_train[df_class0_train.columns[:-1]]
y_train_0=df_class0_train[df_class0_train.columns[-1]]
x_train_1=df_class1_train[df_class1_train.columns[:-1]]
y_train_1=df_class1_train[df_class1_train.columns[:-1]]

# defining likelihood function

def likelihood(x_vector,mean_vector,cov_matrix): # defining likelihood function
    matrix1= np.dot((x_vector-mean_vector).T,np.linalg.inv(cov_matrix))
    power= -0.5*np.dot(matrix1,(x_vector-mean_vector))
    exp_part=np.exp(power)
    # exp_part=bigfloat.exp(pow,bigfloat.precision(100))
    return (exp_part/((2*np.pi)**11.5 * (abs(np.linalg.det(cov_matrix)))**.5))

#defining mean and covariance for both the classes data

mean_0=np.array(x_train_0.mean())
mean_1=np.array(y_train_1.mean())
cov_0=np.cov((x_train_0).T)
cov_1=np.cov((x_train_1).T)

# To store predictions,we define empty list

bayes_prediction=[]

for row in x_test.itertuples(index=False):
        l0=likelihood(np.array(row),mean_0,cov_0)
        l1=likelihood(np.array(row),mean_1,cov_1)
        if l0>l1:
            bayes_prediction.append(0)
        else:
            bayes_prediction.append(1)
            
print()
print()
print("\nBayes Classification Predictions:")
confMatrix=confusion_matrix(y_test, bayes_prediction)
print(confMatrix)
accuracy=accuracy_score(y_test, bayes_prediction)
print("Accuracy",accuracy)
accuracy3=accuracy

ans=pd.DataFrame({'Classification Type':['KNN Classification','Normalizes KNN','Bayes Classification'],'Accuracy':[accuracy1,accuracy2,accuracy3]})
print(ans)