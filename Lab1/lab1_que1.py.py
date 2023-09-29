# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:07:23 2022

@author: sonam
"""

import pandas as pd
df = pd.read_csv(r'D:\IC272 DS3\Lab1/pima-indians-diabetes.csv')
emp_datfram = pd.DataFrame()
for i in df.columns[0:8]:
    emp_datfram = emp_datfram.append({"columnno":i,"mean":df[i].mean(),"median":df[i].median(),"maximum":df[i].max(),"minimum":df[i].min(),"standard deviation":df[i].std()},ignore_index=True)
print(emp_datfram)
print(df[:-1].mode())


