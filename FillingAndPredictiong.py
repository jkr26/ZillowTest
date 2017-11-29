#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import time

"""
Created on Wed Sep  6 19:31:24 2017

@author: jkr
"""


GB_params={'learning_rate':[0.1, 0.2, 0.5], 'n_estimators':[50, 100, 150, 200], 
           'max_depth':[2, 3, 5, 10]}
log_errors=pd.read_csv("~/Documents/Machine Learning/Zillow/train_2016_v2.csv", low_memory=False)
Predictions=pd.DataFrame(columns=['parcelid', 'prediction', 'month', 'year'])
start_time=time.time()
for Data in pd.read_csv("~/Documents/Machine Learning/Zillow/properties_2016.csv",
                 low_memory=False, chunksize=2*10**5):
    HT_mask=Data.hashottuborspa =='nan'
    Data.loc[HT_mask,'hashottuborspa']=0
    Data.loc[~HT_mask,'hashottuborspa']=1
    
    FP_mask=Data.fireplaceflag == 'nan'
    Data.loc[FP_mask, 'fireplaceflag']=0
    Data.loc[~FP_mask, 'fireplaceflag']=1
    
    TD_mask=Data.taxdelinquencyflag == 'nan'
    Data.loc[TD_mask, 'taxdelinquencyflag']=0
    Data.loc[~TD_mask, 'taxdelinquencyflag']=1
    ls=list(Data)
    GB_params={'learning_rate':[0.1, 0.2, 0.5], 'n_estimators':[50, 100, 150, 200], 
           'max_depth':[2, 3, 5, 10]}
    for to_predict in ls:
        msk=Data[to_predict].notnull()
        Non_null_data=Data[msk].copy()
        if (len(Non_null_data)>=10) and (to_predict!='propertycountylandusecode'):
            Non_null_data=Non_null_data.apply(pd.to_numeric, args=('coerce',)).fillna(value=0)
            if len(Non_null_data)<100000 and len(Non_null_data)>100:
                int_msk=np.random.rand(len(Non_null_data))<0.4
                train_y=Non_null_data[to_predict][int_msk].copy()
                test_y=Non_null_data[to_predict][~int_msk].copy()
                del Non_null_data[to_predict]
                a=StandardScaler().fit(Non_null_data)
                Non_null_data=pd.DataFrame(a.transform(Non_null_data), columns=list(Non_null_data))
                int_train=Non_null_data[int_msk]
                int_test=Non_null_data[~int_msk]
                GB_CV=GridSearchCV(GradientBoostingRegressor(), GB_params).fit(int_train, train_y)
                score=GB_CV.score(int_test, test_y)
                print(to_predict+" "+str(score))
                if score>=0:
                    Null_data=Data[~msk].copy()
                    del Null_data[to_predict]
                    Null_data=Null_data.apply(pd.to_numeric, args=('coerce',)).fillna(value=0)
                    Null_data=Null_data.fillna(value=0)
                    Null_data=pd.DataFrame(a.transform(Null_data), columns=list(Null_data))
                    Imputations=GB_CV.predict(Null_data)
                    Data.loc[~msk, to_predict]=Imputations
    Data=Data.apply(pd.to_numeric, args=('coerce',)).fillna(value=0)
    labeled_data=pd.merge(Data, log_errors, how='inner', on='parcelid')
    transaction_dates=labeled_data['transactiondate']
    labeled_data['transactiondate']=pd.to_datetime(labeled_data['transactiondate'])
    labeled_data['month']=labeled_data['transactiondate'].apply(lambda x: x.month)
    labeled_data['year']=labeled_data['transactiondate'].apply(lambda x: x.year)
    del labeled_data['transactiondate']
    Train_data=labeled_data[np.random.rand(len(labeled_data))<.8]
    Test_data=labeled_data[np.random.rand(len(labeled_data))>.8]
    Train_y=Train_data['logerror']
    Test_y=Test_data['logerror']
    del Train_data['logerror']
    del Test_data['logerror']
    IDs=labeled_data['parcelid']
    del Train_data['parcelid']
    del Test_data['parcelid']
    scaler=StandardScaler().fit(Train_data)
    Train_data=pd.DataFrame(scaler.transform(Train_data), columns=list(Train_data))
    Test_data=pd.DataFrame(scaler.transform(Test_data), columns=list(Test_data))
    GB_params={'loss': ['ls', 'lad', 'huber', 'quantile'], 'learning_rate':[0.1, 0.2, 0.5], 'n_estimators':[50, 100, 150, 200], 
           'max_depth':[2, 3, 5, 10]}
    Production_GB=GridSearchCV(GradientBoostingRegressor(), GB_params).fit(Train_data, Train_y)
    score=np.mean(np.abs(Production_GB.predict(Test_data)-Test_y))
    print("Validation MAE is "+str(score))
    Data_IDs=pd.DataFrame(Data['parcelid'])
    del Data['parcelid']
    Data_to_precdict=pd.DataFrame(columns=list(Data))    
    for month in [10, 11, 12]:
        for year in [2016, 2017]:
            Data_to_concat=Data.copy()
            Data_to_concat['month']=month
            Data_to_concat['year']=year
            Data_to_predict=scaler.transform(Data_to_concat)
            preds=Production_GB.predict(Data_to_predict)
            Data_IDs['prediction']=preds
            Data_IDs['month']=[month]*len(Data_IDs)
            Data_IDs['year']=[year]*len(Data_IDs)
    Data_IDs['prediction']=preds
    Predictions=pd.concat([Predictions, Data_IDs])
    
    