# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 18:20:25 2022

@author: sonam
"""
#lab7

#Que-1

"""
steps for PCA :
1. standarize the data
2. Build the covariance matrix
3. Find the Eigenvectors and Eigenvalues
4. Sort the eigenvectors in highest to lowest order and select the number of principal components
"""
#target variable is:Species

#Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#reading the csv file
df=pd.read_csv('D:\IC272 DS3\Lab7\Iris.csv')

"""The iloc() function in python is one of the functions defined in the Pandas module that helps us 
to select a specific row or column from the data set. """
#pandas. dataset.iloc[row, column]
#print(df.iloc[:,0:].describe())

#plotting boxplot for df to check the output labels
# for i in df.columns[:4]:
#    df.boxplot(i,by='Species')
   
X = df.drop(['Species'],axis=1)
Y = df['Species']   
#print(X)
#print(Y)

eigenvalues,eigenvectors= np.linalg.eig(np.cov(X.T))
print(eigenvalues)
#print(eigenvectors)

df_pca = PCA(n_components=2)
df_pca.fit(X)
X_pca = df_pca.transform(X)
#print(X_pca)
df_reduced = pd.DataFrame(X_pca)
df_reduced.rename(columns={0:'REDUCED-1',1:'REDUCED-2'},inplace=True)
#print(df_reduced)

plt.scatter(X_pca[:, 0],X_pca[:, 1])
  
# labeling x and y axes
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()
