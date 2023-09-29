# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 19:27:56 2022

@author: sonam
"""

import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv(r"D:\IC272 DS3\Lab1/pima-indians-diabetes.csv")

#part - a

for i in df.columns[0:7]:
    plt.scatter(df["Age"],df[i])
    plt.title(i + ' vs Age')
    plt.xlabel('Age')
    plt.ylabel(i)
    plt.grid(True)
    plt.show()
    
#part - b

for i in df.columns[0:5]:
    plt.scatter(df["BMI"],df[i])
    plt.title(i + ' vs BMI')
    plt.xlabel('BMI')
    plt.ylabel(i)
    plt.grid(True)
    plt.show()
for i in df.columns[6:8]:
    plt.scatter(df["BMI"],df[i])
    plt.title(i + ' vs BMI')
    plt.xlabel('BMI')
    plt.ylabel(i)
    plt.grid(True)
    plt.show()



