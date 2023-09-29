# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 19:13:05 2022

@author: sonam
"""

#Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

#reading the csv file
df=pd.read_csv('D:\IC272 DS3\Lab7\Iris.csv')

X = df.drop(['Species'],axis=1)
Y = df['Species']   

x_label = [] # Converting the label class as numeric value
for i in range (len(Y)):
    if (Y[i] == 'Iris-setosa'):
        x_label.append(0)
    if (Y[i] == 'Iris-versicolor'):
        x_label.append(1)
    if (Y[i] == 'Iris-virginica'):
        x_label.append(2) 

df_pca = PCA(n_components=2)
df_pca.fit(X)
X_pca = df_pca.transform(X)
#print(X_pca)
df_reduced = pd.DataFrame(X_pca)
df_reduced.rename(columns={0:'REDUCED-1',1:'REDUCED-2'},inplace=True)
#print(df_reduced)

k=[2,3,4,5,6,7]
l=[]

for K in range(2,8):
    # training gaussian mixture model 
    gmm = GaussianMixture(n_components=K)
    gmm.fit(df_reduced)
    
    #predictions from gmm
    labels = gmm.predict(df_reduced)
    #print(labels)
    
    plt.scatter(df_reduced["REDUCED-1"],df_reduced["REDUCED-2"],c=labels,s=50) #s=size of the dot
    plt.show()
    
    print('log likelihood for k = ',K)
    att = X.columns
    l.append((gmm.lower_bound_)*(len(X[att[1]])))
    print((gmm.lower_bound_)*(len(X[att[1]])))
    
    def purity_score(y_true, y_pred):
        #y_true(np.ndarray): n*1 matrix Ground truth labels
        #y_pred(np.ndarray): n*1 matrix Predicted clusters
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
        #print(contingency_matrix)
        # Find optimal one-to-one mapping between cluster labels and true labels
        row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
        # Return cluster accuracy
        return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)
    print('purity score for k=',K)
    print(purity_score(x_label, labels))
    print('     ')
#print(l)
plt.plot(k,l,c='b')
plt.title('K vs log likelihood') 
plt.xlabel('k')
plt.ylabel('log likelihood')
plt.show()
