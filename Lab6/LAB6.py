# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:50:00 2022

@author: sonam
"""

#autoregression:predicting the values of something on the past values of that 
#thing

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg as AR
import warnings
warnings.filterwarnings('ignore')
from statsmodels.graphics.tsaplots import plot_acf

#ques-1

#part-a

df=pd.read_csv('D:\IC272 DS3\Lab6\daily_covid_cases.csv',index_col=0,parse_dates=True)

#creating line plot
X=df.values
df.plot()
plt.xlabel("month and year")
plt.ylabel("Newly confirmed cases")

#part-b

#generating the new time series data with one day lag.
df1=df.shift(-1) #shift used for lagging
df1.plot()
plt.xlabel("month and year")
plt.ylabel("Newly confirmed cases")
plt.show()

#pearrson corellation coffecient between original and laged data.

x1=df['new_cases'].corr(df1['new_cases'])
print("This is the pearrson correlation coffecient between two data(1)   :",x1 )

#part-c

plt.scatter(df['new_cases'],df1['new_cases'])
plt.title('scatter plot b/w original an one day lag data')
plt.show()

#yes it approximately matches the correlation coffiecient in 1.b by seeing the 
#picture both the graph follow the same line

#part-d

#shift-1

df1.plot()
plt.xlabel("month and year")
plt.ylabel("Newly confirmed cases")
plt.show()
x=df['new_cases'].corr(df1['new_cases'])
print("This is the pearrson correlation coffecient between two data (1)  :",x )

#shift-2

df2=df.shift(-2)
df2.plot()
plt.xlabel("month and year")
plt.ylabel("Newly confirmed cases")
plt.show()
#pearrson corellation coffecient between original and laged data(2).
x2=df['new_cases'].corr(df2['new_cases'])
print("This is the pearrson correlation coffecient between two data(2)   :",x2 )

#shift-3

df3=df.shift(-3)
df3.plot()
plt.xlabel("month and year")
plt.ylabel("Newly confirmed cases")
#pearrson corellation coffecient between original and laged data(3).
x3=df['new_cases'].corr(df3['new_cases'])
print("This is the pearrson correlation coffecient between two data(3)   :",x3)

#shift-4

df4=df.shift(-4)
df4.plot()
plt.xlabel("month and year")
plt.ylabel("Newly confirmed cases")
#pearrson corellation coffecient between original and laged data(4).
x4=df['new_cases'].corr(df4['new_cases'])
print("This is the pearrson correlation coffecient between two data(4)   :",x4)

#shift-5

df5=df.shift(-5)
df5.plot()
plt.xlabel("month and year")
plt.ylabel("Newly confirmed cases")
#pearrson corellation coffecient between original and laged data(5).
x5=df['new_cases'].corr(df5['new_cases'])
print("This is the pearrson correlation coffecient between two data(3)   :",x5)

#shift-6

df6=df.shift(-6)
df6.plot()
plt.xlabel("month and year")
plt.ylabel("Newly confirmed cases")
#pearrson corellation coffecient between original and laged data(6).
x6=df['new_cases'].corr(df6['new_cases'])
print("This is the pearrson correlation coffecient between two data(6)   :",x6 )

t1=np.array([0.999064414471503,0.996375107483936,0.9919388573086093,0.9857861789952966,0.9779675657688203,0.9685303994656104])
t2= np.array([1,2,3,4,5,6] )
plt.plot(t1,t2,color ="red")
plt.xlabel("correlation")
plt.ylabel("lagged values")
plt.show()

#part-e
plot_acf(t2)
plt.show()

#ques-2

#part-a

y=df["new_cases"]
x=df.drop("new_cases",axis=1)
test_size = 0.35 
X = df.values
#The math.ceil function rounds a number up to the nearest integer or to the 
#nearest multiple of specified significance.
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

def auto_reg_coefficients(train_data,test_data,lag=1):      # function for returning the list of coefficients
    model = AR(train_data, lags=lag)
    model_fit = model.fit() 
    coef = model_fit.params 
    return coef

def auto_reg_predictions(train_data,test_data,lag=1):       #function for returning the prediction array
    model = AR(train_data, lags=lag)
    model_fit = model.fit() 
    coef = model_fit.params 

    hist = train_data[(len(train_data)-lag):]
    hist = [hist[i] for i in range(len(hist))]

    pred=[]
    for j in range(len(test_data)):
        lth=len(hist)
        lagg = [hist[i] for i in range(lth-lag,lth)]
        w_0=coef[0]
        for d in range(lag):
            w_0 += coef[d+1] * lagg[lag-d-1]
        obs=test_data[j]
        pred.append(w_0)
        hist.append(obs)
    return pred

for i in range(1,6):
    print("@ Coefficient for p = ",i," is ")
    print(auto_reg_coefficients(train,test,i))
    
#part-b

for i in range(1,6):
    print("@ Plot , RMSE , MAPE for p = ",i," is :")
    pre_ref=auto_reg_predictions(train,test,i)
    plt.scatter(test,pre_ref)
    plt.plot(test,pre_ref,color="r")
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.show()
    
    rmse=0
    s=0
    mape=0
    for j in range(len(test)):
        rmse+=((test[j].item() - pre_ref[j].item())**2)
        s+=test[j].item()
        mape+=abs(test[j].item() - pre_ref[j].item())/test[j].item()
        
    print("RMSE -",((rmse/len(test))**(0.5))/(s/len(test)))
    print("MAPE -",mape/len(test))

#ques-3

#Mean Absolute Percentage Error (MAPE) can be used in machine learning to measure 
#the accuracy of a model.
#The difference between them is, MAPE measures the deviation from the actual data 
#in terms of percentage, that is the only difference between them. The similarity 
#between them is they both measure the absolute error. So in both the negative and 
#the positive errors, cancel out each other. RMSE method is more accurate.

l=[1,5,10,15,25]
l_rmse=[]
l_mape=[]
for i in l:
    pre_ref=auto_reg_predictions(train,test,i)
    rmse=0
    mape=0
    s=0
    for j in range(len(test)):
        rmse+=((test[j].item() - pre_ref[j].item())**2)
        s+=test[j].item()
        mape+=abs(test[j].item() - pre_ref[j].item())/test[j].item()
        
    l_rmse.append(((rmse/len(test))**(0.5))/(s/len(test)))
    l_mape.append(mape/len(test))
    
plt.bar(l,l_rmse)
plt.title("Plot for RMSE vs lag values")
plt.xlabel("lag values")
plt.ylabel("RMSE")
plt.show()

plt.bar(l,l_mape)
plt.title("Plot for mape vs lag values")
plt.xlabel("lag values")
plt.ylabel("mape")
plt.show()

i=1
print((2/(len(train))**(0.5)))
train=pd.DataFrame(train)
while True:
    df_lag=train.shift(i)
    co=train[0].corr(df_lag[0])
    if (abs(co))>(2/(len(train))**(0.5)):
        print(co)
        print(i)
    else:
        break
    i+=1

for i in range(100):
    df_lag=train.shift(i)
    co=train[0].corr(df_lag[0])
    if (abs(co))>(2/(len(train))**(0.5)):
        print(co)
        print(i)

#ques-4

l=[]
l_rmse=[]
l_mape=[]
i=1
train=pd.DataFrame(train)
while True:
    train=pd.DataFrame(train)
    df_lag=train.shift(i)
    co=train[0].corr(df_lag[0])
    if (abs(co))>(2/(len(train))**(0.5)):
        l.append(i)
        train=train.values
        pre_ref=auto_reg_predictions(train,test,i)
        rmse=0
        mape=0
        s=0
        print(i)
        for j in range(len(test)):
            rmse+=((test[j].item() - pre_ref[j].item())**2)
            s+=test[j].item()
            mape+=abs(test[j].item() - pre_ref[j].item())/test[j].item()

        l_rmse.append(((rmse/len(test))**(0.5))/(s/len(test)))
        l_mape.append(mape/len(test))
    else:
        break
    i+=1

plt.plot(l,l_rmse)
plt.title("Plot for RMSE vs lag values")
plt.xlabel("lag values")
plt.ylabel("RMSE")
plt.show()

plt.plot(l,l_mape)
plt.title("Plot for mape vs lag values")
plt.xlabel("lag values")
plt.ylabel("MAPE")
plt.show()
