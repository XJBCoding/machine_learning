#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 12:32:56 2018

@author: yuanjihuang
"""
import math
from scipy.io import loadmat
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def ordinary_least_squares(x,y,test_x,test_y):
    reg = linear_model.LinearRegression()
    reg.fit(x,y)
    #print(reg.coef_)
    return mean_squared_error(reg.predict(test_x),test_y)
    


def sparse_linear_predictor(x,y,test_x,test_y):
    combination_list = []
    
    total_combination = int(math.factorial(len(x[0]))/(math.factorial(3)*math.factorial(len(x[0])-3)))
    def dfs(choose,x,y,num,start,count):
        if count == total_combination:
            return 
        
        if num == 3:
            count += 1
            #print(choose,num,start)
            combination_list.append(choose.copy())
            return
        else:
            for i in range(start,len(choose)):
                choose[i] = 1
                dfs(choose,x,y,num+1,i+1,count)
                choose[i] = 0
            return
    dfs(np.array([0] * len(x[0])),x,y,0,1,0)
    #print(combination_list)
    min_err = float('inf')
    min_reg = linear_model.LinearRegression()
    table1 = []
    table2 = []
    v_list = []
    for current_combination in combination_list:
        
        current_combination[0] = 1
        #print(current_combination)
        new_x = []
        new_test_x = []
        for i in range(len(current_combination)):
            if (current_combination[i]):
                new_x.append(x[:,i])
                new_test_x.append(test_x[:,i])
        new_x = np.array(new_x).transpose()
        new_test_x = np.array(new_test_x).transpose()
        reg = linear_model.LinearRegression()
        reg.fit(new_x,y)
        cur_err = mean_squared_error(reg.predict(new_x),y)
        if cur_err < min_err:
            min_reg = reg
            min_err = cur_err
            v_list = current_combination    
        table1.append(cur_err)
        table2.append(reg.coef_)
    res_list = []
    for i in range(len(v_list)):
        if v_list[i]:
            res_list.append(i)
    #print(min_reg.coef_)
    
    return (min_reg, mean_squared_error(min_reg.predict(new_test_x),test_y), res_list)
        
def correlated_variable(x,v_list):
    corr_matrix = np.corrcoef(x,rowvar = 0)
    res = []
    for v in v_list:
        sort_list = np.argsort(abs(corr_matrix[:,v]))
        first = sort_list[-3]
        first_name = name_list[first-1]
        first_corr = corr_matrix[:,v][first]
        second = sort_list[-4]
        second_name = name_list[second-1]
        second_corr = corr_matrix[:,v][second]
        res.append([(first_name,first_corr),(second_name,second_corr)])
    return res
if __name__ == '__main__':
    name_list = ['fixed acidity','volatile acidity','citric acid',
                 'residual sugar','chlorides','free sulfur dioxide',
                 'total sulfur dioxide','density','pH','sulphates','alcohol']
    wine = loadmat('wine.mat')
    test_risk1 = ordinary_least_squares(wine['data'],wine['labels'],wine['testdata'],wine['testlabels'])
    reg,test_risk2,v_list = sparse_linear_predictor(wine['data'],wine['labels'],wine['testdata'],wine['testlabels'])
    
    cor_list = correlated_variable(wine['testdata'],v_list)
    #print(cor_list)
  