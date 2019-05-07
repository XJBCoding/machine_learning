#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 22:43:22 2018

@author: yuanjihuang
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
pi = math.pi
z = np.linspace(-pi,pi,1001)
#plt.plot(z,y)
table = np.array([[0] * 1000] * 20,dtype = 'float')
for l in range(1000):
    e = np.random.normal(0, 1, 1001)
    for k in range(1,21):
        y = np.sin(3*z/2)
        x = np.array([[1 for _ in range(k+1)] for _ in range(len(z))],dtype='float')
        for i in range(1,k+1):
            x[:,i] = np.power(z,i)
        reg = linear_model.LinearRegression()
        reg.fit(x,y)
        table[k-1,l] = mean_squared_error(reg.predict(x),y)
    #print(l)    
means = np.mean(table,axis = 1)
plt.xticks([i for i in range(1,21)])
plt.xlabel("k")
plt.ylabel("average risk")
plt.plot([i for i in range(1,21)],means)