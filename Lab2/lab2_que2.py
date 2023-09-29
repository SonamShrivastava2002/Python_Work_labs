# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 09:38:41 2022

@author: sonam
"""
import pandas as pd
df = pd.read_csv(r'D:\IC272 DS3\Lab2\landslide_data3_miss.csv')

# part - a

print("total number of tuples deleted from stationid: ",df["stationid"].isnull().sum())
df.dropna(subset = ["stationid"],inplace = True)
df.reset_index(inplace = True)
print(df)

# part - b

pre_row = df.shape[0]
df.dropna(thresh=(2/3*(df.columns.size)+1),inplace=True) 
df.reset_index(drop = True)
new_row = df.shape[0]
print(df)
print("the total no. of rows deleted",pre_row-new_row)
