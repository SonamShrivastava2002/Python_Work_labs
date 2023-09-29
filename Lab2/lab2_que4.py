# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 16:27:05 2022

@author: sonam
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
df_miss = pd.read_csv(r'D:\IC272 DS3\Lab2\landslide_data3_miss.csv')
df_origin = pd.read_csv(r'D:\IC272 DS3\Lab2\landslide_data3_original.csv')

#part - a

#replacing missing values with mean values
df_miss.fillna(df_miss.mean(),inplace = True)
print(df_miss)

#part(a)-i

#comparing the original data with missing values data having missing values replaced with mean values
df_miss.fillna(df_miss.mean(),inplace = True)
emp_dataframe1 = pd.DataFrame()
emp_dataframe2 = pd.DataFrame()
emp_dataframe3 = pd.DataFrame()
for i in df_miss.columns[2:9]:
    emp_dataframe1 = emp_dataframe1.append({"column name":i,"mean(original)":df_origin[i].mean(),"mean(missing)":df_miss[i].mean()},ignore_index=True)
    emp_dataframe2 = emp_dataframe2.append({"column name":i,"median(original)":df_origin[i].median(),"median(missing)":df_miss[i].median()},ignore_index=True)
    emp_dataframe3  = emp_dataframe3.append({"column name":i,"standard deviation(original)":df_origin[i].std(),"standard deviation(missing)":df_miss[i].std()},ignore_index=True)
print(emp_dataframe1)
print(emp_dataframe2)
print(emp_dataframe3)
print(df_miss.mode())
print(df_origin.mode()) 

#part(a) - ii

#calculating RMSE for original and missing value data
df_miss.fillna(df_miss.mean(),inplace = True)
list1 = []
attributes = ['temperature','humidity','pressure','rain','lightavgw/o0','lightmax','moisture']
for i in df_origin.columns[2:9]:
    for j in i:
        squ = np.square(np.subtract(df_origin[i],df_miss[i])).mean()
        squ = math.sqrt(squ)
    list1.append(squ)
data_frame = pd.DataFrame()
data_frame['attributes'] = attributes
data_frame['RMSE']= list1
print(data_frame)
plt.plot(data_frame["attributes"],data_frame["RMSE"])
plt.show()


#part - b

#replacing the missing values with interpolated values
df_miss = df_miss.interpolate()
print(df_miss)

#part(b)-i

#comparing the original data with missing values data having missing values replaced with interpolated values
df_miss = df_miss.interpolate()
emp_dataframe1 = pd.DataFrame()
emp_dataframe2 = pd.DataFrame()
emp_dataframe3 = pd.DataFrame()
for i in df_miss.columns[2:9]:
    emp_dataframe1 = emp_dataframe1.append({"column name":i,"mean(original)":df_origin[i].mean(),"mean(missing)":df_miss[i].mean()},ignore_index=True)
    emp_dataframe2 = emp_dataframe2.append({"column name":i,"median(original)":df_origin[i].median(),"median(missing)":df_miss[i].median()},ignore_index=True)
    emp_dataframe3  = emp_dataframe3.append({"column name":i,"standard deviation(original)":df_origin[i].std(),"standard deviation(missing)":df_miss[i].std()},ignore_index=True)
print(emp_dataframe1)
print(emp_dataframe2)
print(emp_dataframe3)
print(df_miss.mode())
print(df_origin.mode()) 

#part(b) - ii

#calculating RMSE for original and missing value data
df1 = pd.read_csv(r'D:\IC272 DS3\labcsvfiles\landslide_data3_original.csv')
df_miss = df_miss.interpolate()
list1 = []
attributes = ['temperature','humidity','pressure','rain','lightavgw/o0','lightmax','moisture']
for i in df_origin.columns[2:9]:
    for j in i:
        squ = np.square(np.subtract(df_origin[i],df_miss[i])).mean()
        squ = math.sqrt(squ)
    list1.append(squ)
data_frame = pd.DataFrame()
data_frame['attributes'] = attributes
data_frame['RMSE']= list1
print(data_frame)
plt.plot(data_frame["attributes"],data_frame["RMSE"])
plt.show()
