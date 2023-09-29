# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 13:30:33 2022

@author: sonam
"""

import pandas as pd
df = pd.read_csv(r'D:\IC272 DS3\Lab1/pima-indians-diabetes.csv')
data_frame = pd.DataFrame({'pregs':df["pregs"],'skin':df["skin"]})
data_frame.hist()
