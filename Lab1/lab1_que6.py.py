# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 14:43:46 2022

@author: sonam
"""

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r'D:\IC272 DS3\Lab1/pima-indians-diabetes.csv')
for i in df.columns[0:8]:
    plt.boxplot(df[i])
    plt.title(i)
    plt.grid()
    plt.show()
    