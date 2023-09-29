# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 10:56:55 2022

@author: sonam
"""

import pandas as pd
import numpy as np
df = pd.read_csv(r'D:\IC272 DS3\labcsvfiles\pima-indians-diabetes.csv')

#que1

#replacing outliers to median of the attributes
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
print(df)

#part a
x_max = df[i].max()
x_min = df[i].min()
print("the min and max before normalization of",i)
print("        ")
print("max:",x_max)
print("min:",x_min)
print("        ")
for j in df[i]:
    normal = ((j-x_min)/(x_max-x_min))*7 + 5
    list2.append(normal)
    #print(normal)
print("the min and max after nomalization of",i)
print("        ")
print("max:",max(list2))
print("min:",min(list2))
list2.clear()

#part b
x_mean = df[i].mean()
x_std = df[i].std()
print("mean and standard deviation of ",i,"before standarization")
print("        ")
print("mean:",x_mean)
print("standard deviation:",x_std)
print("     ")
for j in df[i]:
      normal = (j-x_mean)/x_std
      df[i].replace(to_replace = j,value = normal,inplace=True)
print("mean and standard deviation of",i,"after standarization")
print("      ")      
print("mean:",df[i].mean())
print("std:",df[i].std())
print("     ")
print(df)
    