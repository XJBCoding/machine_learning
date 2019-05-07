#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 21:51:47 2018

@author: yuanjihuang
"""
from scipy.io import loadmat
import numpy as np
from numpy import exp,dot,array,zeros,ones,log,mean
import matplotlib.pyplot as plt


MAX_ITERATION = 10000



def objective(x,b,b0,y):
    return np.mean(np.log(1 + exp(b0 + x.dot(b))) - y * (b0 + x.dot(b)))


def gradient_descent(data,label):
    d = data.shape[1]
    b0 = 0
    b = zeros((d, 1))
    for k in range(1,MAX_ITERATION):
        #take derivative
        tem1 = exp(b0 + dot(data,b)) / (1 + exp(b0 + dot(data,b)))
        grad_b = mean(data * tem1, axis=0) - mean(data * label, axis=0)
        grad_b0 = mean(exp(b0 + dot(data,b)) / (1 + exp(b0 + dot(data,b))) - label)
        grad_b = grad_b.reshape((d,1))  
   
        a = 1.0
        while objective(data, b - a * grad_b, b0 - a * grad_b0, label)\
        > objective(data,b,b0,label)- (a/2 * (grad_b0 ** 2 + dot(grad_b.flatten(),grad_b.flatten()))):
            a /= 2
        b = b - a * grad_b
        b0 = b0 - a * grad_b0
        print(k,objective(data,b,b0,label))
        if(objective(data,b,b0,label) < 0.65064):
            return k
        

def compute_error_rate(b,b0,data,label):
    predict = 1/(1+exp(-b0-data.dot(b))) > 0.5
    error = predict != label
    return sum(sum(error)) / data.shape[0]
    
def gradient_descent2(data,label,test_data,test_label):
    d = data.shape[1]
    b0 = 0
    b = zeros((d, 1))
    best_err = 1
    for k in range(1,MAX_ITERATION):
        #take derivative
        tem1 = exp(b0 + dot(data,b)) / (1 + exp(b0 + dot(data,b)))
        grad_b = mean(data * tem1, axis=0) - mean(data * label, axis=0)
        grad_b0 = mean(exp(b0 + dot(data,b)) / (1 + exp(b0 + dot(data,b))) - label)
        grad_b = grad_b.reshape((d,1))  
   
        a = 1.0
        while objective(data, b - a * grad_b, b0 - a * grad_b0, label)\
        > objective(data,b,b0,label)- (a/2 * (grad_b0 ** 2 + dot(grad_b.flatten(),grad_b.flatten()))):
            a /= 2
        b = b - a * grad_b
        b0 = b0 - a * grad_b0
        print(k,objective(data,b,b0,label))
        if (k & k - 1) == 0:
            err = compute_error_rate(b,b0,test_data,test_label)
            if (err >= 0.99 * best_err) and (k >= 32):
                return (k,objective(data,b,b0,label),err)
            elif err < best_err:
                best_err = err
            print(best_err)



if __name__ == "__main__":
    logreg = loadmat('logreg.mat')
    data = logreg['data']
    labels = logreg['labels']
    result = gradient_descent(data,labels)
    A = array([[1,0,0],[0,20,0],[0,0,1]])
    tran_data = dot(A,data.transpose()).transpose()
    result2 = gradient_descent(tran_data,labels)
    
    train_x = data[:3276]
    train_y = labels[:3276]
    test_x = data[3276:]
    test_y = labels[3276:]
    
    (result3,value3,err3) = gradient_descent2(train_x,train_y,test_x,test_y)
    
    tran_train_x = tran_data[:3276]
    tran_test_x = tran_data[3276:]
    (result4,value4,err4) = gradient_descent2(tran_train_x,train_y,tran_test_x,test_y)
