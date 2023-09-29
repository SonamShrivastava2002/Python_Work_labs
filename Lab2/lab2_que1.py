# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 08:06:47 2022

@author: sonam
"""

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r'D:\IC272 DS3\Lab2\landslide_data3_miss.csv')
miss_value = df.isnull().sum()
print(miss_value)
plt.plot(miss_value)
plt.grid()
plt.show()
