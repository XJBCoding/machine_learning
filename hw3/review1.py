#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:48:18 2018

@author: yuanjihuang
"""
import math
import csv
import numpy as np
from scipy import sparse
from random import sample

def load_data(text,row):
    res_x = []
    res_y = np.empty([row-1,],dtype='int')
    with open(text) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for r in csv_reader:
            #print(r[0],r[1])
            if line_count == 0:
                line_count += 1
            else:
                res_x.append(r[1])
                res_y[line_count-1] = 1 if int(r[0]) == 1 else -1
                #print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                line_count += 1

        print(line_count)
        return (res_x,res_y)


def construct_dic(text):
    log_idf = {}
    dic = {}
    mapping = {}
    mapping['affine parameter'] = 0
    log_idf['affine parameter'] = 0
    pos = 1
    for row in text:
        for word in row.split(' '):
            if word in dic.keys():
                dic[word] += 1
            else:
                mapping[word] = pos
                dic[word] = 1
                pos += 1
    D = len(text)
    for item in dic.items():
        log_idf[item[0]] = math.log10(D / item[1])
    return (log_idf,mapping)

def construct_mi(text,y):
    mi = {}
    P_c0 = 0
    P_c1 = 0
    P_word = {}
    N = len(text)
    dic = {}
    pos_dic = {}
    neg_dic = {}
    mapping = {}
    
    mapping['affine parameter'] = 0
    pos = 1
    
    for i in range(len(text)):
        if y[i] == 1:
            P_c1 += 1
            for word in text[i].split(' '):
                if word in dic.keys():
                    dic[word] += 1
                else:
                    mapping[word] = pos
                    dic[word] = 1
                    pos += 1
                if word in pos_dic.keys():
                    pos_dic[word] += 1
                else:
                    pos_dic[word] = 1
                if word not in neg_dic.keys():
                    neg_dic[word] = 0
        else:
            P_c0 += 1
            for word in text[i].split(' '):
                if word in dic.keys():
                    dic[word] += 1
                else:
                    mapping[word] = pos
                    dic[word] = 1
                    pos += 1
                if word in neg_dic.keys():
                    neg_dic[word] += 1
                else:
                    neg_dic[word] = 1
                if word not in pos_dic.keys():
                    pos_dic[word] = 0
                    
                    
    pos_dic['affine parameter'] = P_c1
    neg_dic['affine parameter'] = P_c0
    dic['affine parameter'] = N
    
    for key in dic.keys():
        dic[key] /= N
    for key in pos_dic.keys():
        pos_dic[key] /= P_c1
    for key in neg_dic.keys():
        neg_dic[key] /= P_c0
    P_c0 /= N
    P_c1 /= N
    
    mi = {}
    for key in dic.keys():
        print(neg_dic[key]/dic[key],pos_dic[key]/dic[key])
        if neg_dic[key]/dic[key] > 0:
            tem1 = (P_c0*math.log10(neg_dic[key]/dic[key]))
        else:
            tem1 = 0
        if pos_dic[key]/dic[key] > 0:
            tem2 = (P_c1*math.log10(pos_dic[key]/dic[key]))
        else:
            tem2 = 0
            
        mi[key] = tem1+tem2
    
    return (mi,mapping)


def construct_bimapping(text):
    mapping = {}
    mapping['affine parameter bi'] = 0
    pos = 1
    for row in text:
        words = row.split(' ')
        for i in range(len(words)-1):
            combination = words[i]+' '+words[i+1]
            if combination in mapping.keys():
                continue
            else:
                mapping[combination] = pos
                pos += 1
                
    return mapping


def save_dic(dic,name):
    w = csv.writer(open(name, "w"))
    for key, val in dic.items():
        w.writerow([key, val])
        
        
def load_dic(name):
    dic = {}
    with open(name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for r in csv_reader:
            dic[r[0]] = int(r[1])
    return dic


def unigram_feature(x,y,mapping):
    print('start constructing feature')
    l = len(mapping)  
    r = []
    c = []
    data = []
    for i,row in enumerate(x):
        if not i % 10000:    print(i)
        tem_x = {}
        for word in row.split(' '):
            if word in tem_x.keys():
                tem_x[word] += 1
            else:
                tem_x[word] = 1
        tem_x['affine parameter'] = 1
        for item in tem_x.items():
            #print(i,mapping[item[0]],item[1])
            if item[0] in mapping.keys():
                r.append(i)
                c.append(mapping[item[0]])
                data.append(item[1])
    res = sparse.csr_matrix((data,(r,c)),shape = [len(x),l])
    return res
    
        
def bigram_feature(x,y,bimapping):
    print('start constructing feature')
    l = len(bimapping)  
    r = []
    c = []
    data = []
    for i,row in enumerate(x):
        if not i % 10000:    print(i)
        tem_x = {}
        words = row.split(' ')
        for j in range(len(words)):
            if j < len(words)-1:
                combination = words[j]+' '+words[j+1]
                if combination in tem_x.keys():
                    tem_x[combination] += 1
                else:
                    tem_x[combination] = 1
                word = words[j]
                if word in tem_x.keys():
                    tem_x[word] += 1
                else:
                    tem_x[word] = 1
            else:
                word = words[j]
                if word in tem_x.keys():
                    tem_x[word] += 1
                else:
                    tem_x[word] = 1
                    
        tem_x['affine parameter bi'] = 1
        for item in tem_x.items():
            #print(i,bimapping[item[0]],item[1])
            if item[0] in bimapping.keys():
                r.append(i)
                c.append(bimapping[item[0]])
                data.append(item[1])
    res = sparse.csr_matrix((data,(r,c)),shape = [len(x),l])
    return res


def tfidf_feature(log_idf,x,y,mapping):
    print('start constructing feature')
    l = len(mapping)  
    r = []
    c = []
    data = []
    for i,row in enumerate(x):
        if not i % 10000:    print(i)
        tem_x = {}
        for word in row.split(' '):
            if word in tem_x.keys():
                tem_x[word] += 1
            else:
                tem_x[word] = 1
        tem_x['affine parameter'] = 1
        for item in tem_x.items():
            #print(i,mapping[item[0]],item[1])
            if item[0] in mapping.keys():
                r.append(i)
                c.append(mapping[item[0]])
                data.append(item[1] * log_idf[item[0]])
    res = sparse.csr_matrix((data,(r,c)),shape = [len(x),l])
    return res

def mi_feature(mi,x,y,mapping):
    print('start constructing feature')
    l = len(mapping)  
    r = []
    c = []
    data = []
    for i,row in enumerate(x):
        if not i % 10000:    print(i)
        tem_x = {}
        for word in row.split(' '):
            if word in tem_x.keys():
                tem_x[word] += 1
            else:
                tem_x[word] = 1
        tem_x['affine parameter'] = 1
        for item in tem_x.items():
            #print(i,mapping[item[0]],item[1])
            if item[0] in mapping.keys():
                r.append(i)
                c.append(mapping[item[0]])
                data.append(item[1] * mi[item[0]])
    res = sparse.csr_matrix((data,(r,c)),shape = [len(x),l])
    return res


def online_perceptron(feature,y,mapping):
    print('start training')
    n = feature.shape[0]
    w_list = np.zeros([n,len(mapping)],dtype = 'float')
    w = np.zeros([len(mapping),1],dtype = 'float')
    w_final = np.zeros([len(mapping),1],dtype = 'float')
    #pass 1 and 2
    for j in range(2):
        pos = 0
        sel = sample(range(n),n)
        shuffle_feature = feature[sel]
        shuffle_y = y[sel]
        for i in range(shuffle_feature.shape[0]):
            if not i % 1000:    print(i)
            #print(float(shuffle_feature[i] @ w) * shuffle_y[i]  <= 0)
            if float(shuffle_feature[i] @ w) * shuffle_y[i]  <= 0:
                w = w + (shuffle_y[i] * shuffle_feature[i]).transpose()
            if j == 1:
                w_final += w/n
            pos += 1
            
    return w_final
    
def predict(w,x,y):
    print('start prdicting')
    res = np.ones([x.shape[0],1],dtype = 'int')
    tem = x @ w > 0
    res = res * tem
    tem2 = res == 0
    return res-tem2

def err_rate(predict,y):
    return np.sum(predict != y.reshape(predict.shape)) / predict.shape[0]
#%% loading data
if __name__ == '__main__':
    train_x,train_y = load_data(r'/Users/yuanjihuang/Documents/machine learning/hw3/reviews_tr.csv',1000001)
    test_x,test_y = load_data(r'/Users/yuanjihuang/Documents/machine learning/hw3/reviews_te.csv',320123)
    tem_train_x = train_x[:100]
    tem_train_y = train_y[:100]
    log_idf,mapping,= construct_dic(tem_train_x)
    
#%% save dictionary
    #save_dic(dic,'dic.csv')
    #save_dic(mapping,'map.csv')
    
#%% import dictionary
    #dic = load_dic('dic.csv')
    #mapping = load_dic('map.csv')
#%% constructing feature
    
#%% saving feature
    #sparse.save_npz('feature1.npz', feature1)    
#%% loading feature    
    #feature1 = sparse.load_npz('feature1.npz')
#%% unigram
    train_unigram = unigram_feature(tem_train_x,tem_train_y,mapping)
    test_unigram = unigram_feature(test_x,test_y,mapping)
    w_final_unigram = online_perceptron(train_unigram,tem_train_y,mapping)
    train_predict_unigram = predict(w_final_unigram,train_unigram,tem_train_y)
    test_predict_unigram = predict(w_final_unigram,test_unigram,test_y)
    print('unigram train error rate:',err_rate(train_predict_unigram,tem_train_y))
    print('unigram test error rate:',err_rate(test_predict_unigram,test_y))
#%% tfidf
    train_tfidf = tfidf_feature(log_idf,tem_train_x,tem_train_y,mapping)
    test_tfidf = tfidf_feature(log_idf,test_x,test_y,mapping)
    w_final_tfidf = online_perceptron(train_tfidf,tem_train_y,mapping)
    train_predict_tfidf = predict(w_final_tfidf,train_tfidf,tem_train_y)
    test_predict_tfidf = predict(w_final_tfidf,test_tfidf,test_y)
    print('tfidf train error rate:',err_rate(train_predict_tfidf,tem_train_y))
    print('tfidf test error rate:',err_rate(test_predict_tfidf,test_y))
#%%
    bimapping = construct_bimapping(tem_train_x)
    bimapping.update(mapping)
    train_bigram = bigram_feature(tem_train_x,tem_train_y,bimapping)
    test_bigram = bigram_feature(test_x,test_y,bimapping)
    w_final_bigram = online_perceptron(train_bigram,tem_train_y,bimapping)
    train_predict_bigram = predict(w_final_bigram,train_bigram,tem_train_y)
    test_predict_bigram = predict(w_final_bigram,test_bigram,test_y)
    print('bigram train error rate:',err_rate(train_predict_bigram,tem_train_y))
    print('bigram test error rate:',err_rate(test_predict_bigram,test_y))
#%%
    mi,mapping = construct_mi(tem_train_x,tem_train_y)
    train_mi = mi_feature(mi,tem_train_x,tem_train_y,mapping)
    test_mi = mi_feature(mi,test_x,test_y,mapping)
    w_final_mi = online_perceptron(train_mi,tem_train_y,mapping)
    train_predict_mi = predict(w_final_mi,train_mi,tem_train_y)
    test_predict_mi = predict(w_final_mi,test_mi,test_y)
    print('mi train error rate:',err_rate(train_predict_mi,tem_train_y))
    print('mi test error rate:',err_rate(test_predict_mi,test_y))






    