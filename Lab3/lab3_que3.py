# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 20:59:42 2022

@author: sonam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv(r'D:\IC272 DS3\labcsvfiles\pima-indians-diabetes.csv')
df = df.drop('class',axis=1)

#part-a
list1 = []
list2 = []
for i in df.columns[:-1]:
      for j in df[i]:
          q1 = np.percentile(df[i],[25])
          q3 = np.percentile(df[i],[75])
          iqr = q3-q1
          upper_limit = q3 + 1.5*iqr
          lower_limit = q1 - 1.5*iqr
          if (lower_limit > j or j > upper_limit):
              list1.append(j)
      #printing outliers in a list with their medians
      # print(i,list1)
      # print(df[i].median())
     
      for k in range(len(list1)):
          df[i].replace(to_replace = list1[k],value = df[i].median(),inplace=True)
      list1.clear()
x_mean = df[i].mean()
x_std = df[i].std()
for j in df[i]:
      normal = (j-x_mean)/x_std
      df[i].replace(to_replace = j,value = normal,inplace=True)
      pca = PCA(n_components=2)
      red_df = pca.fit_transform(df)
      
final_df = pd.DataFrame(red_df,columns = ['col1','col2'])
cova = np.cov(final_df.T)
eig_value = np.linalg.eig(cova)
#print(eig_value)
plt.scatter(final_df['col1'],final_df['col2'])
plt.show()

#part-b
df = np.array(df)
cova = np.cov(df.T)
eig_value,eig_vec = np.linalg.eig(cova)
sort_list = sorted(eig_value,reverse=True)
plt.plot([i for i in range(8)], sort_list)
plt.show()

#part - c
Error=[]
N=len(df)
for i in range(2,9,1):
    pca=PCA(n_components=i)
    pca.fit(df)
    x_pca = pca.transform(df)
    X_ori = pca.inverse_transform(x_pca)
    X_error = 0
    for j in range(N):
        sum = 0
        for k in range(8):
            sum += (X_ori[j][k]-df[j][k])**2
        X_error += sum**0.5
    Error.append(X_error)
plt.plot([i for i in range(2,9,1)],Error)
plt.xlabel("No. of componenets (l)")
plt.ylabel("Reconstruction Error")
plt.title("Plot of recontruction error vs No. of Components")
plt.show()
    
#part - d
pca=PCA(n_components=8)
pca.fit(df)
trans=pca.transform(df)
Cov=np.cov(df,rowvar = False)
print(Cov)
