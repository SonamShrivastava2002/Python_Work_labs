# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 00:42:42 2022

@author: sonam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv(r'D:\IC272 DS3\labcsvfiles\pima-indians-diabetes.csv')

#part a
mean = [0,0]
cov = [[13,-3],[-3,5]]
x,y  = np.random.multivariate_normal(mean,cov,1000).T
plt.scatter(x,y)
#print(x,y)
plt.show()

#part b
cov = [[13,-3],[-3,5]]
eig_value,eig_vec = np.linalg.eig(cov)
print("eigen values for cov matrix:",eig_value)
print("eigen vectors for cov matrix:",eig_vec)

#plotting
mean = [0,0]
cov = [[13,-3],[-3,5]]
x,y  = np.random.multivariate_normal(mean,cov,1000).T

x_pos = [0,0]
y_pos = [0,0]
x_dir = [2,5.5]
y_dir = [3,-4]
plt.scatter(x,y,marker='x')
plt.title('plot of 2d synthetic data and eigen directions')
plt.quiver(x_pos,y_pos,x_dir,y_dir,scale=30)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

#part - c
D = np.random.multivariate_normal(mean,cov,1000).T
eig_vec1 = eig_vec[:,0]
eig_vec2 = eig_vec[:,1]
rng = np.random.RandomState()
D = np.random.multivariate_normal(mean,cov,1000)
D_transformed1 = np.dot(D, eig_vec1)
D_transformed2 = np.dot(D, eig_vec2)
plt.plot(D_transformed2)
plt.plot(D_transformed1)
plt.show()

#part - d
pca = PCA(n_components=2)
pca.fit(df)
df_pca = pca.transform(df)
print("transformed shape:", df_pca.shape)
print(df_pca.var())
D=pca.inverse_transform(df_pca)
print(D.shape)
print('Reconstructional Error:',(((D-df)**2).sum()/len(D)))
