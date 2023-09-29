# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 16:15:35 2022

@author: sonam
"""

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r'D:\IC272 DS3\Lab1/pima-indians-diabetes.csv')
data_frame = pd.DataFrame({'pregs':df["pregs"]})
grp_y = data_frame.groupby(df["class"])
data_frame1= grp_y.get_group(0)
data_frame1.hist()
plt.title('pregs for class 0')
data_frame1=grp_y.get_group(1)
data_frame1.hist()
plt.title('pregs for class 1')
