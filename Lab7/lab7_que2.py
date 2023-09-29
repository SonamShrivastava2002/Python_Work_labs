# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 14:33:17 2022

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
x_label = [] # Converting the label class as numeric value
for i in range (len(Y)):
    if (Y[i] == 'Iris-setosa'):
        x_label.append(0)
    if (Y[i] == 'Iris-versicolor'):
        x_label.append(1)
    if (Y[i] == 'Iris-virginica'):
        x_label.append(2)  
#print(X)
#print(Y)

df_pca = PCA(n_components=2)
df_pca.fit(X)
X_pca = df_pca.transform(X)
#print(X_pca)
df_reduced = pd.DataFrame(X_pca)
df_reduced.rename(columns={0:'REDUCED-1',1:'REDUCED-2'},inplace=True)
#print(df_reduced)

# Step 1 and 2 - Choose the number of clusters (k) and select random centroid for each cluster
k=3
# Select random observation as centroids
centroids =df_reduced.sample(n=3)
plt.scatter(df_reduced["REDUCED-1"],df_reduced["REDUCED-2"])
plt.scatter(centroids["REDUCED-1"],centroids["REDUCED-2"],c='black',s=100)
plt.xlabel('reduced column 1')
plt.ylabel('reduced column 2')
plt.show()

#part - a

kmeans = KMeans(n_clusters=3).fit(df_reduced)
y_kmeans = kmeans.predict(df_reduced)
plt.scatter(df_reduced["REDUCED-1"],df_reduced["REDUCED-2"], c=y_kmeans, s=50)
plt.scatter(0.66443351,-0.33029221,c='green',marker='*',s=200)
plt.scatter(-2.64084076,0.19051995,c='red',marker='*',s=200)
plt.scatter(2.34645113,0.27235455,c='blue',marker='*',s=200)
#print(kmeans.cluster_centers_)

labels = kmeans.labels_
#print(labels)

#part - b
print(kmeans.inertia_)

#part - c

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

print(purity_score(x_label, labels))
