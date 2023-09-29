# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 18:52:20 2022

@author: sonam
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\IC272 DS3\Lab2\landslide_data3_miss.csv')

#part-a

df = df.interpolate()
print(df)

#plotting boxplots

#temperature
plt.boxplot(df["temperature"])
plt.title('boxplot for temperature')
plt.grid()
plt.show()

#rain
plt.boxplot(df["rain"])
plt.title('boxplot for rain')
plt.grid()
plt.show()

#outliers for temperature
list_temp = []
q1_temp = np.percentile(df['temperature'],[25])
q3_temp = np.percentile(df['temperature'],[75])
iqr_temp = q3_temp-q1_temp
upper_temp = q1_temp - (1.5)*iqr_temp
lower_temp = q3_temp + (1.5)*iqr_temp
for i in df['temperature']:
    if (upper_temp>=i or i>=lower_temp):
        list_temp.append(i)
print(list_temp)

#outliers for rain
list_rain = []
q1_rain = np.percentile(df['rain'],[25],interpolation='midpoint')
q3_rain = np.percentile(df['rain'],[75],interpolation='midpoint')
iqr_rain = q3_rain-q1_rain
upper_rain = q1_rain - (1.5)*iqr_rain
lower_rain = q3_rain + (1.5)*iqr_rain
for i in df['rain']:
    if (upper_rain>=i or i>=lower_rain):
        list_rain.append(i)
print(list_rain)

#part-b

#replacing outliers for temperature
list_temp = []
q1_temp = np.percentile(df['temperature'],[25])
q3_temp = np.percentile(df['temperature'],[75])
iqr_temp = q3_temp-q1_temp
upper_temp = q1_temp - (1.5)*iqr_temp
lower_temp = q3_temp + (1.5)*iqr_temp
for i in df['temperature']:
    if (upper_temp>=i or i>=lower_temp):
        list_temp.append(i)
for j in range(len(list_temp)):
    df.replace(to_replace = list_temp[j],value = df['temperature'].median())
print(df)

#replacing outliers for rain
list_rain = []
q1_rain = np.percentile(df['rain'],[25],interpolation='midpoint')
q3_rain = np.percentile(df['rain'],[75],interpolation='midpoint')
iqr_rain = q3_rain-q1_rain
upper_rain = q1_rain - (1.5)*iqr_rain
lower_rain = q3_rain + (1.5)*iqr_rain
for i in df['rain']:
    if (upper_rain>=i or i>=lower_rain):
        list_rain.append(i)
for j in range(len(list_rain)):
    df.replace(to_replace = list_rain[j],value = df['rain'].median())
print(df)

#after replacing plotting the boxplots

#temperature
list_temp = []
q1_temp = np.percentile(df['temperature'],[25])
q3_temp = np.percentile(df['temperature'],[75])
iqr_temp = q3_temp-q1_temp
upper_temp = q1_temp - (1.5)*iqr_temp
lower_temp = q3_temp + (1.5)*iqr_temp
for i in df['temperature']:
    if (upper_temp>=i or i>=lower_temp):
        list_temp.append(i)
for j in range(len(list_temp)):
    df=df.replace(to_replace = list_temp[j],value = df['temperature'].median())
print(df)
plt.boxplot(df["temperature"])
plt.title('boxplot for temperature')
plt.grid()
plt.show()

#rain
list_rain = []
q1_rain = np.percentile(df['rain'],[25],interpolation='midpoint')
q3_rain = np.percentile(df['rain'],[75],interpolation='midpoint')
iqr_rain = q3_rain-q1_rain
upper_rain = q1_rain - (1.5)*iqr_rain
lower_rain = q3_rain + (1.5)*iqr_rain
for i in df['rain']:
    if (upper_rain>=i or i>=lower_rain):
        list_rain.append(i)
for j in range(len(list_rain)):
    df=df.replace(to_replace = list_rain[j],value = df['rain'].median())
print(df)
plt.boxplot(df["rain"])
plt.title('boxplot for rain')
plt.grid()
plt.show()
