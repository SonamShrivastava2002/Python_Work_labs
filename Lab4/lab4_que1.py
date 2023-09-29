# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 19:02:31 2022

@author: sonam
"""

#importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

#reading csv file
df = pd.read_csv('D:\IC272 DS3\Lab4\SteelPlateFaults-2class.csv')

#spliting the data of each row
y = df[df['Class']==0]
x = df[df['Class']==1]
x_label = x['Class']
y_label = y['Class']

# Using the command train_test_split from scikit-learn to split the data into 
# train data and test data. Train data contain 70% of tuples from each of the
# class and test data contain remaining 30% of tuples from each class. 

X_train_1, X_test_1, X_label_train_1, X_label_test_1 = train_test_split(x, x_label, test_size=0.3, random_state=104,shuffle=True)
X_train_0, X_test_0, X_label_train_0, X_label_test_0 = train_test_split(y, y_label, test_size=0.3, random_state=104,shuffle=True)

X_train=pd.concat([X_train_0, X_train_1])
X_train.reset_index(drop=True,inplace=True)
X_test=pd.concat([X_test_0, X_test_1])
X_test.reset_index(drop=True,inplace=True)
X_label_train=pd.concat([X_label_train_0, X_label_train_1])
X_label_test=pd.concat([X_label_test_0, X_label_test_1])

print(X_train)
print(X_test)
#Saving the train data as SteelPlateFaults-train.csv and saving the test data 
#as SteelPlateFaultstest.csv
X_train.to_csv('D:\IC272 DS3\Lab4\SteelPlateFaults-train.csv')
X_test.to_csv('D:\IC272 DS3\Lab4\SteelPlateFaults-test.csv')

X_train.drop('Class', axis=1, inplace=True)
X_test.drop('Class', axis=1, inplace=True)

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

#part - b

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
      