#!/usr/bin/env python

from __future__ import print_function
from scipy.io import loadmat
import numpy as np

def estimate_naive_bayes_classifier(X,Y):# X: data Y: labels
    #raise Exception("IMPLEMENT ME")
    y = np.unique(Y).shape[0]
    row = X.shape[0]
    
    params = {}
    params['pi'] = np.array([0] * y, dtype='float64')
    params['mu'] = np.array([[0] * d] * y, dtype='float64')
    for i in range(y):
        label = i + 1
        indices = Y == label
        cur_X = X[indices,:]
        params['pi'][i] = cur_X.shape[0] / row
        params['mu'][i] = (1 + np.sum(cur_X,axis = 0)) / (2 + cur_X.shape[0])
    return params
         
def predict(params,X):
    res = np.array([0] * X.shape[0])
    part1 = np.log(params['pi'] * np.prod(1 - params['mu'],axis = 1))
    part1 = np.tile(part1, (X.shape[0],1)).transpose()
    part2 = np.log(params['mu'] / (1 - params['mu']))
    part2 = part2.dot(X.toarray().transpose())
    res = part1 + part2
    res = np.argmax(res,axis = 0) + 1
    return res

def print_top_words(params,vocab):
    part2 = np.log(params['mu'] / (1 - params['mu']))
    a = part2[1] - part2[0]
    sort_a = np.argsort(a)
    neg = sort_a[:20]
    pos = sort_a[-20:]
    pos_vocab = []
    neg_vocab = []
    for i in range(20):
        pos_vocab.append(vocab[pos[i]])
    for i in range(20):
        neg_vocab.append(vocab[neg[i]])
    print('top postive 20 words:',pos_vocab)
    print('top negtive 20 words:',neg_vocab)
    #raise Exception("IMPLEMENT ME")
    return 

def load_data():
    return loadmat('news.mat')

def load_vocab():
    with open('news.vocab') as f:
        vocab = [ x.strip() for x in f.readlines() ]
    return vocab


if __name__ == '__main__':
    news = load_data()
    
    # 20-way classification problem
    d = 61188
    
    data = news['data']
    labels = news['labels'][:,0]
    testdata = news['testdata']
    testlabels = news['testlabels'][:,0]
    
    params1 = estimate_naive_bayes_classifier(data,labels)

    pred = predict(params1,data) # predictions on training data
    testpred = predict(params1,testdata) # predictions on test data

    print('20 classes: training error rate: %g' % np.mean(pred != labels))
    print('20 classes: test error rate: %g' % np.mean(testpred != testlabels))

    # binary classification problem

    indices = (labels==1) | (labels==16) | (labels==20) | (labels==17) | (labels==18) | (labels==19)
    data2 = data[indices,:]
    labels2 = labels[indices]
    labels2[(labels2==1) | (labels2==16) | (labels2==20)] = 1
    labels2[(labels2==17) | (labels2==18) | (labels2==19)] = 2
    testindices = (testlabels==1) | (testlabels==16) | (testlabels==20) | (testlabels==17) | (testlabels==18) | (testlabels==19)
    testdata2 = testdata[testindices,:]
    testlabels2 = testlabels[testindices]
    testlabels2[(testlabels2==1) | (testlabels2==16) | (testlabels2==20)] = 1
    testlabels2[(testlabels2==17) | (testlabels2==18) | (testlabels2==19)] = 2

    params2 = estimate_naive_bayes_classifier(data2,labels2)
    pred2 = predict(params2,data2) # predictions on training data
    testpred2 = predict(params2,testdata2) # predictions on test data

    print('2 classes: training error rate: %g' % np.mean(pred2 != labels2))
    print('2 classes: test error rate: %g' % np.mean(testpred2 != testlabels2))

    vocab = load_vocab()
#%%
    print_top_words(params2,vocab)

