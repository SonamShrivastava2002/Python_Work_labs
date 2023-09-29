# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 19:17:57 2022

@author: sonam
"""

#DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
#eps: it is the radius of specific neighborhoods. If the distance between two points 
#is less than or equal to esp, it will be considered its neighbors.minPts: minimum 
#number of data points in a given neighborhood to form the clusters. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

#reading the csv file
df=pd.read_csv('D:\IC272 DS3\Lab7\Iris.csv')

"""The iloc() function in python is one of the functions defined in the Pandas module that helps us 
to select a specific row or column from the data set. """
#pd.dataset.iloc[row, column]
print(df.iloc[:,0:].describe())

#plotting boxplot for df to check the output labels
for i in df.columns[:4]:
    df.boxplot(i,by='Species')
   
X = df.drop(['Species'],axis=1)
Y = df['Species']   
#print(X)
#print(Y)
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

#For different combination of eps and min_samples plotting the number of clusters

#part-a

for i in range(1,4):
    print(i)
    for j in range(4,11):
        print(j)
        dbscan_model=DBSCAN(eps=i, min_samples=j).fit(df_reduced)
        DBSCAN_predictions = dbscan_model.labels_
        #print(DBSCAN_predictions)
        
        plt.scatter(df_reduced["REDUCED-1"],df_reduced["REDUCED-2"],c=DBSCAN_predictions,s=50)
        plt.show()
        
#part - b
        
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
        print('purity score for i,j=',i,',',j,)
        print(purity_score(x_label, DBSCAN_predictions))
        print('     ')
   