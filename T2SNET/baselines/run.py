
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import numpy
import math
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn import metrics

import time
# import dataframe_image as dfi
import pandas as pd

import Setting
from myfunctions import lr_model,svr_model,ann_model,rf_model,lstm_model,hybrid_ceemdan_rf,hybrid_ceemdan_lstm,proposed_method

url_univdorm= '/Users/zouguojian/RF-LSTM-CEEMDAN/Dataset/UnivDorm_Prince.csv'
univdorm= pd.read_csv(url_univdorm)
data_univdorm= univdorm[(univdorm['timestamp'] > '2015-03-01') & (univdorm['timestamp'] < '2015-06-01')]
dfs_univdorm=data_univdorm['energy']
datas_univdorm=pd.DataFrame(dfs_univdorm)

hours=Setting.n_hours
data_partition=Setting.data_partition
max_features=Setting.max_features
epoch=Setting.epoch
batch_size=Setting.batch_size
neuron=Setting.neuron
lr=Setting.lr
optimizer=Setting.optimizer

start_time = time.time()
proposed_method_univdorm=proposed_method(dfs_univdorm,hours,data_partition,max_features,epoch,batch_size,neuron,lr,optimizer)
proposed_method_time_univdorm=time.time() - start_time
print("--- %s seconds - proposed_method- univdorm ---" % (proposed_method_time_univdorm))
print(proposed_method_univdorm[0:3])

'''
#Linear Regression

start_time = time.time()
lr_univdorm=lr_model(datas_univdorm,hours,data_partition)
lr_time_univdorm=time.time() - start_time
print("--- %s seconds - Linear Regression- univdorm ---" % (lr_time_univdorm))

#Support Vector Regression
start_time = time.time()
svr_univdorm=svr_model(datas_univdorm,hours,data_partition)
svr_time_univdorm=time.time() - start_time
print("--- %s seconds - Support Vector Regression- univdorm ---" % (svr_time_univdorm))


#ANN
start_time = time.time()
ann_univdorm=ann_model(datas_univdorm,hours,data_partition)
ann_time_univdorm=time.time() - start_time
print("--- %s seconds - ANN- univdorm ---" % (ann_time_univdorm))

#random forest
start_time = time.time()
rf_univdorm=rf_model(datas_univdorm,hours,data_partition,max_features)
rf_time_univdorm=time.time() - start_time
print("--- %s seconds - Random Forest- univdorm ---" % (rf_time_univdorm))

#LSTM
start_time = time.time()
lstm_univdorm=lstm_model(datas_univdorm,hours,data_partition,max_features,epoch,batch_size,neuron,lr,optimizer)
lstm_time_univdorm=time.time() - start_time
print("--- %s seconds - lstm- univdorm ---" % (lstm_time_univdorm))


#CEEMDAN RF
start_time = time.time()
ceemdan_rf_univdorm=hybrid_ceemdan_rf(dfs_univdorm,hours,data_partition,max_features)
ceemdan_rf_time_univdorm=time.time() - start_time
print("--- %s seconds - ceemdan_rf- univdorm ---" % (ceemdan_rf_time_univdorm))

#CEEMDAN LSTM
start_time = time.time()
ceemdan_lstm_univdorm=hybrid_ceemdan_lstm(dfs_univdorm,hours,data_partition,max_features,epoch,batch_size,neuron,lr,optimizer)
ceemdan_lstm_time_univdorm=time.time() - start_time
print("--- %s seconds - ceemdan_lstm- univdorm ---" % (ceemdan_lstm_time_univdorm))


#proposed method
start_time = time.time()
proposed_method_univdorm=proposed_method(dfs_univdorm,hours,data_partition,max_features,epoch,batch_size,neuron,lr,optimizer)
proposed_method_time_univdorm=time.time() - start_time
print("--- %s seconds - proposed_method- univdorm ---" % (proposed_method_time_univdorm))

running_time_univdorm=pd.DataFrame([lr_time_univdorm,svr_time_univdorm,ann_time_univdorm,
                                   rf_time_univdorm,lstm_time_univdorm,ceemdan_rf_time_univdorm,
                                   ceemdan_lstm_time_univdorm,proposed_method_time_univdorm])
running_time_univdorm=running_time_univdorm.T
running_time_univdorm.columns=['LR','SVR','ANN','RF','LSTM','CEEMDAN RF','CEEMDAN LSTM','Proposed Method']
proposed_method_univdorm_df=proposed_method_univdorm[0:3]
result_univdorm=pd.DataFrame([lr_univdorm,svr_univdorm,ann_univdorm,rf_univdorm,lstm_univdorm,ceemdan_rf_univdorm,
                    ceemdan_lstm_univdorm,proposed_method_univdorm_df])
result_univdorm=result_univdorm.T
result_univdorm.columns=['LR','SVR','ANN','RF','LSTM','CEEMDAN RF','CEEMDAN LSTM','Proposed Method']
univdorm_summary=pd.concat([result_univdorm,running_time_univdorm],axis=0)

univdorm_summary.set_axis(['MAPE(%)', 'RMSE','MAE','running time (s)'], axis='index')

univdorm_summary.style.set_caption("University Dormitory Results")
index = univdorm_summary.index
index.name = "university dormitory results"
print(univdorm_summary.to_string())
'''