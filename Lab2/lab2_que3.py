# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:46:24 2022

@author: sonam
"""

import pandas as pd
df = pd.read_csv(r'D:\IC272 DS3\Lab2\landslide_data3_miss.csv')
df.dropna(thresh = (8/3),inplace = True)
df.reset_index(inplace = True)
print("total number of tuples deleted:",df.isnull().sum())
print("total number of missing values:",df.isnull().sum().sum())
