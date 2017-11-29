# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import time
start_time=time.time()
Data=pd.read_csv("~/Documents/Machine Learning/Zillow/Train_1.csv", low_memory=False)

Data_filled_na=Data#.fillna(value=0)

to_del=['transactiondate','Unnamed: 0','Unnamed: 0.1','latitude','longitude','propertyzoningdesc', 'parcelid']

for string in to_del:
    del Data_filled_na[string]
HT_mask=Data_filled_na.hashottuborspa =='nan'
Data_filled_na.loc[HT_mask,'hashottuborspa']=0
Data_filled_na.loc[~HT_mask,'hashottuborspa']=1

FP_mask=Data_filled_na.fireplaceflag == 'nan'
Data_filled_na.loc[FP_mask, 'fireplaceflag']=0
Data_filled_na.loc[~FP_mask, 'fireplaceflag']=1

TD_mask=Data_filled_na.taxdelinquencyflag == 'nan'
Data_filled_na.loc[TD_mask, 'taxdelinquencyflag']=0
Data_filled_na.loc[~TD_mask, 'taxdelinquencyflag']=1


ls=list(Data_filled_na)


GB_params={'learning_rate':[0.1, 0.2, 0.5], 'n_estimators':[50, 100, 150, 200], 
           'max_depth':[2, 3, 5, 10]}

scores={}
Not_enough_data={}
for to_predict in ls:
    Non_null_data=Data[Data[to_predict].notnull()].copy()
    if (len(Non_null_data)>=10) and (to_predict!='propertycountylandusecode') and (to_predict!='logerror'):
        Non_null_data=Non_null_data.fillna(value=0)
        Non_null_data=Non_null_data.astype(str).convert_objects(convert_numeric=True)
        int_msk=np.random.rand(len(Non_null_data))<0.5
        train_y=Non_null_data[to_predict][int_msk].copy()
        test_y=Non_null_data[to_predict][~int_msk].copy()
        del Non_null_data[to_predict]
        Non_null_data=Non_null_data.fillna(value=0)
        a=StandardScaler().fit(Non_null_data)
        Non_null_data=a.transform(Non_null_data)
        int_train=Non_null_data[int_msk]
        int_test=Non_null_data[~int_msk]
        GB_CV=GridSearchCV(GradientBoostingRegressor(), GB_params).fit(int_train, train_y)
        score=GB_CV.score(int_test, test_y)
        if score>=0.1:
            TP_mask=Data[to_predict].notnull()
            Null_data=Data[~TP_mask].copy()
            del Null_data[to_predict]
            Null_data=Null_data.fillna(value=0)
            Null_data=Null_data.astype(str).convert_objects(convert_numeric=True)
            Null_data=Null_data.fillna(value=0)
            Non_null_data=a.transform(Non_null_data)
            Data_filled_na.loc[TP_mask, to_predict]=GB_CV.predict(Non_null_data)
        print(str(score))
        scores[to_predict]=score
    else:
        Not_enough_data[to_predict]=len(Non_null_data)

Data_filled_na=Data_filled_na.astype(str).convert_objects(convert_numeric=True)
Response=Data_filled_na['logerror'].copy()
del Data_filled_na['logerror']
Data_filled_na=Data_filled_na.fillna(value=0)
Normalized_data=StandardScaler().fit_transform(Data_filled_na)
msk=np.random.rand(len(Data_filled_na))
#train=Normalized_data[msk<0.8]
#test=Normalized_data[msk>=0.8]
train=Data_filled_na[msk<0.4]
test=Data_filled_na[msk>=0.4]

train_y=Response[msk<0.4]
test_y=Response[msk>=0.4]

#Bucket_response=

GB_params={'loss': ['ls', 'lad', 'huber', 'quantile'], 'learning_rate':[0.1, 0.2, 0.5], 'n_estimators':[50, 100, 150, 200], 
           'max_depth':[2, 3, 5, 10]}
gb_reg=GridSearchCV(GradientBoostingRegressor(), GB_params).fit(train, train_y)

gb_score=gb_reg.score(test, test_y)
print("--- %s seconds---" %(time.time()-start_time))