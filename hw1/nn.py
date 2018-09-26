#!/usr/bin/env python

from __future__ import print_function
from scipy.io import loadmat
from random import sample
import numpy as np
import time    
import matplotlib.pyplot as plt

def nn(X,Y,test):
    #raise Exception("IMPLEMENT ME")
    m = test.shape[0]
    X_T = X.transpose()
    matrix_1 = np.matrix(np.sum(test**2, axis=1))
    matrix_1 = np.tile(matrix_1.transpose(), (1,n))
    matrix_2 = np.matrix(np.sum(X**2, axis=1))
    matrix_2 = np.tile(matrix_2, (m, 1))
    matrix_3 = np.array([[0] * n] * m,dtype = 'float')
    matrix_3 = test.dot(X_T)
    dis = matrix_1 + matrix_2 - 2 * matrix_3
    nearest = dis.argmin(axis = 1)
    
    res = Y[nearest.transpose()]
    return res

if __name__ == '__main__':
    ocr = loadmat('ocr.mat')
    num_trials = 10
    test_err_list = []
    err_std_list = []
    n_list = []
    for n in [ 1000, 2000, 4000, 8000 ]:
        test_err = np.zeros(num_trials)
        for trial in range(num_trials):
            sel = sample(range(len(ocr['data'].astype('float'))),n)
            start = time.time()
            preds = nn(ocr['data'].astype('float')[sel], ocr['labels'][sel], ocr['testdata'].astype('float'))
            end = time.time()
            print(n,' time:',end-start)
            test_err[trial] = np.mean(preds != ocr['testlabels'])
        print("%d\t%g\t%g" % (n,np.mean(test_err),np.std(test_err)))
        err_std_list.append(np.std(test_err))
        test_err_list.append(np.mean(test_err))
        n_list.append(n) 
    '''
    plt.figure()
    plt.errorbar(n_list, test_err_list, yerr=err_std_list)
    plt.xlabel("n")
    plt.ylabel("test error rate")
    plt.title("Learning Curve")
    '''
