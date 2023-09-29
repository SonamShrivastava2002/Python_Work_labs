# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 19:34:53 2022

@author: sonam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

df = pd.read_csv('D:\IC272 DS3\Lab7\Iris.csv')

#reduced the data

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

# Step 1 and 2 - Choose the number of clusters (k) and select random centroid for each cluster
k=[2,3,4,5,6,7]
l=[]
for K in range(2,8):
    # Select random observation as centroids
    centroids =df_reduced.sample(n=K)
    plt.scatter(df_reduced["REDUCED-1"],df_reduced["REDUCED-2"])
    plt.scatter(centroids["REDUCED-1"],centroids["REDUCED-2"],c='black')
    plt.xlabel('reduced column 1')
    plt.ylabel('reduced column 2')
    plt.show()

    kmeans = KMeans(n_clusters=K).fit(df_reduced)
    y_kmeans = kmeans.predict(df_reduced)
    plt.scatter(df_reduced["REDUCED-1"],df_reduced["REDUCED-2"], c=y_kmeans, s=50, )
    plt.show()
    #print('centroids for k=',K)
    #print(kmeans.cluster_centers_)
    labels = kmeans.labels_
    #print(labels)

    print('distortion for k = ',K)
    l.append(kmeans.inertia_)
    print(kmeans.inertia_)
    
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
plt.title('K vs distortion') 
plt.xlabel('k')
plt.ylabel('distortion')
plt.show()
    
    
    

    
    
    
    
    
    
    
    
    
    
    