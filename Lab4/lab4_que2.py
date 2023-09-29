# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:06:45 2022

@author: sonam
"""

#importing libraries

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

#reading csv file and resetting index

df = pd.read_csv('D:\IC272 DS3\Lab4\SteelPlateFaults-train.csv')
df.reset_index(drop=True,inplace=True)

#Normalizing all the attributes (except class attribute) for SteelPlateFaults-train
list1 = []
#print(df)
for i in df.columns[:-1]:
    x_max = df[i].max()
    x_min = df[i].min()
    for j in df[i]:
        normal = (j-x_min)/(x_max-x_min)
        list1.append(normal)
        df[i].replace(to_replace = j,value = normal,inplace=True)
    list1.clear()

#Saving the file

df.to_csv('D:\IC272 DS3\Lab4\SteelPlateFaults-train-Normalised.csv')
print(df)

#reading normalised csv file
df2 = pd.read_csv('D:\IC272 DS3\Lab4\SteelPlateFaults-train-Normalised.csv')

#spliting the data of each row

y = df2.TypeOfSteel_A300
x = df2.drop('TypeOfSteel_A300',axis = 1)

#Using the command train_test_split from scikit-learn to split the data into 
# train data and test data. Train data contain 70% of tuples of TypeOfSteel_A300 
# and test data contain remaining 30% of tuples of TypeOfSteel_A300. 

X_train, X_test, X_label_train, X_label_test = train_test_split(x, y, test_size=0.3, random_state=104,shuffle=True)

# classifying every test tuple using K-nearest neighbor (KNN) method for the 
# different values of K=1, 3, and 5. 

#for k=1

classifier1 = KNeighborsClassifier(n_neighbors=1)
classifier1.fit(X_train,  X_label_train)
X_pred1 = classifier1.predict(X_test)
print(X_pred1)

#for k=3

classifier2 = KNeighborsClassifier(n_neighbors=3)
classifier2.fit(X_train,  X_label_train)
X_pred2 = classifier2.predict(X_test)
print(X_pred2)

#for k =5

classifier3 = KNeighborsClassifier(n_neighbors=5)
classifier3.fit(X_train,  X_label_train)
X_pred3 = classifier3.predict(X_test)
print(X_pred3)

#part - a

#Find confusion matrix for each K.

#for k = 1

confusion_matrix1 = confusion_matrix(X_label_test, X_pred1)
print(confusion_matrix1)

#for k = 3

confusion_matrix2 = confusion_matrix(X_label_test, X_pred2)
print(confusion_matrix2)

#for k = 5

confusion_matrix3 = confusion_matrix(X_label_test, X_pred3)
print(confusion_matrix3)

#part-b 

#Finding the classification accuracy for each K.

#for k = 1

accuracy1 = accuracy_score(X_label_test,X_pred1)*100
print('Accuracy of the model:' + str(round(accuracy1, 1)) + ' %.')

#for k = 3

accuracy2 = accuracy_score(X_label_test,X_pred2)*100
print('Accuracy of the model:' + str(round(accuracy2, 3)) + ' %.')

#for k = 5

accuracy3 = accuracy_score(X_label_test,X_pred3)*100
print('Accuracy of the model:' + str(round(accuracy3, 5)) + ' %.')

# repeating the same for SteelPlateFaults-test using the minimum and maximum 
# values of train dataset

#reading csv file and resetting index

df = pd.read_csv('D:\IC272 DS3\Lab4\SteelPlateFaults-train.csv')
df1 = pd.read_csv('D:\IC272 DS3\Lab4SteelPlateFaults-test.csv')
df.reset_index(drop=True,inplace=True)
df1.reset_index(drop=True,inplace=True)

#Normalizing all the attributes (except class attribute) for SteelPlateFaults-test

list2 = []
for i in df1.columns[:-1]:
    x_max = df[i].max()
    x_min = df[i].min()
    for j in df1[i]:
        normal = (j-x_min)/(x_max-x_min)
        list2.append(normal)
        df1[i].replace(to_replace = j,value = normal,inplace=True)
    list2.clear()

#Saving the file

df1.to_csv('D:\IC272 DS3\Lab4\SteelPlateFaults-test-Normalised.csv')
df3 = pd.read_csv('D:\IC272 DS3\Lab4\SteelPlateFaults-test-Normalised.csv')
# print(df3)

#spliting the data of each row
y = df3.TypeOfSteel_A300
x = df3.drop('TypeOfSteel_A300',axis = 1)
X_train, X_test, X_label_train, X_label_test = train_test_split(x, y, test_size=0.3, random_state=104,shuffle=True)

#classifying every test tuple using K-nearest neighbor (KNN) method for the 
#different values of K=1, 3, and 5. 

#for k=1

classifier1 = KNeighborsClassifier(n_neighbors=1)
classifier1.fit(X_train,  X_label_train)
X_pred1 = classifier1.predict(X_test)
print(X_pred1)

#for k=3

classifier2 = KNeighborsClassifier(n_neighbors=3)
classifier2.fit(X_train,  X_label_train)
X_pred2 = classifier2.predict(X_test)
print(X_pred2)

#for k =5

classifier3 = KNeighborsClassifier(n_neighbors=5)
classifier3.fit(X_train,  X_label_train)
X_pred3 = classifier3.predict(X_test)
print(X_pred3)

#part - a

#Find confusion matrix for each K.

#for k = 1

confusion_matrix1 = confusion_matrix(X_label_test, X_pred1)
print(confusion_matrix1)

#for k = 3

confusion_matrix2 = confusion_matrix(X_label_test, X_pred2)
print(confusion_matrix2)

#for k = 5

confusion_matrix3 = confusion_matrix(X_label_test, X_pred3)
print(confusion_matrix3)

#Finding the classification accuracy for each K.

#for k = 1

accuracy1 = accuracy_score(X_label_test,X_pred1)*100
print('Accuracy of the model:' + str(round(accuracy1, 1)) + ' %.')

#for k = 3

accuracy2 = accuracy_score(X_label_test,X_pred2)*100
print('Accuracy of the model:' + str(round(accuracy2, 3)) + ' %.')

#for k = 5

accuracy3 = accuracy_score(X_label_test,X_pred3)*100
print('Accuracy of the model:' + str(round(accuracy3, 5)) + ' %.')