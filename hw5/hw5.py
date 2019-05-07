# -*- coding: utf-8 -*-
import csv
import os
import sys
import numpy as np
from keras.utils import to_categorical
from keras.layers import Dense, Input, Dropout
from keras.models import Model,Sequential
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#import javalang
import random
def load_data(text,row,col):
    res_x = np.empty([row,col],dtype='float')
    res_y = np.zeros([row,1],dtype='float')
    with open(text) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for r in csv_reader:
            #print(r[0],r[1])
            if line_count == 0:
                line_count += 1
            else:
                for i in range(len(r)):
                    if i == 0:
                        res_y[line_count-1] = (float(r[i])-1922)/(2010-1922)
                        #res_y[line_count-1] = r[i]
                    else:
                        res_x[line_count-1][i-1] = r[i]
                line_count += 1
            if line_count % 5000 == 0:
                print(line_count)
        return (res_x,res_y)

def load_test_data(text,row,col):
    res_x = np.empty([row,col],dtype='float')
    with open(text) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for r in csv_reader:
            #print(r[0],r[1])
            if line_count == 0:
                line_count += 1
            else:
                for i in range(len(r)):
                    res_x[line_count-1][i] = r[i]
                line_count += 1
            if line_count % 5000 == 0:
                print(line_count)
        return res_x
def mapping_back(y):
    return np.round(y * (2010-1922) + 1922)
#%% feature_selection
if __name__ == "__main__":
    x,y = load_data('data.csv',463715,90)
    test_x = load_test_data('testdata.csv',51630,90)
    #%% feature_selection
    feature_num = 90
    pca = PCA(n_components = feature_num)
    x_pca = pca.fit_transform(x)
    test_x_pca = pca.transform(test_x)
    tem_x = x_pca
    #tem_x = x
    test_x_pca = np.array(test_x_pca)
    tem_y = y
      
    #%%    
    VALIDATION_SPLIT = 0.1
    indices = np.arange(tem_x.shape[0])
    np.random.shuffle(indices)
    tem_x = tem_x[indices]
    tem_y = tem_y[indices]    
    num_validation_samples = int(VALIDATION_SPLIT * tem_x.shape[0])
    train_x = tem_x[num_validation_samples:]
    train_y = tem_y[num_validation_samples:]
    val_x = tem_x[:num_validation_samples]
    val_y = tem_y[:num_validation_samples]
    
    
    #%%
    #network
    model = Sequential()
    model.add(Dense(units = feature_num, activation = 'relu', input_dim = feature_num))
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dense(units = 1))
    model.compile(loss='mean_absolute_error',optimizer='adam',metrics=[metrics.mae])
    model.fit(train_x,train_y,batch_size = 600,epochs = 30,validation_data = (val_x,val_y))
    
    #%%
    result = model.predict(val_x)
    result = mapping_back(result)
    val_y = tem_y[:num_validation_samples]
    val_y = mapping_back(val_y)
    error = np.mean(abs(result-val_y))
    res = model.predict(test_x_pca)
    res = mapping_back(res)
    #%%
    with open('submit.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['id','label'])
        for i in range(len(res)):
            writer.writerow([i+1,int(res[i][0])])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    