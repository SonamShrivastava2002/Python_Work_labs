# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 23:41:05 2022

@author: sonam
"""

import pandas as pd
df = pd.read_csv(r'D:\IC272 DS3\Lab1/pima-indians-diabetes.csv')

# part a
correlation_Age=pd.DataFrame()
for i in df.columns[:-1]:
      correlation_Age=correlation_Age.append({"coloum name":i,"correlation coefficient":df[i]. corr(df["Age"])},ignore_index=True)
print(correlation_Age)

#part b

correlation_BMI=pd.DataFrame()
for i in df.columns[:-1]:
      correlation_BMI=correlation_BMI.append({"coloum name":i,"correlation coefficient":df[i]. corr(df["BMI"])},ignore_index=True)
print(correlation_BMI) 
  