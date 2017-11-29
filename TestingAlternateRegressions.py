#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import time
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from keras.layers import Input, Dense, normalization, Dropout
from keras.layers.merge import Add
from keras.models import Model
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")
GB_params={'loss': ['ls', 'lad', 'huber', 'quantile'], 'learning_rate':[0.1, 0.2, 0.5], 'n_estimators':[50, 100, 150, 200], 
       'max_depth':[2, 3, 5, 10]}
KR_params={'kernel':['linear', 'polynomial', 'rbf'], 'degree':[2, 3, 4]}
"""
Created on Wed Sep  6 19:31:24 2017

@author: jkr
"""
def unsupervised_preprocess(data):
    data_dict={}
    data_dict['Raw']=data.copy()
    data_dict['Whitened']=pd.DataFrame(StandardScaler().fit_transform(data.copy()), columns=list(data))
    for n in [5, 10, 20, 30]:
        data_dict['PCA with {0}'.format(n)]=PCA(n_components=n).fit_transform(data)
        data_dict['FA with {0}'.format(n)]=FactorAnalysis(n_components=n).fit_transform(data)
    return data_dict
        
def functional_regression_model(input_dim, width, depth, skip_length, prob):
    starting_point=0
    inputs=Input(shape=(input_dim,))
    x=normalization.BatchNormalization()(inputs)
    x=Dense(width, activation='relu')(inputs)
    x=normalization.BatchNormalization()(x)
    for k in range(depth):
        if k%skip_length==0:
            if starting_point!=0:
                x=Add()([x, starting_point])
            starting_point=x
        x=Dense(width, activation='relu')(x)
        x=Dropout(prob)(x)
        x=normalization.BatchNormalization()(x)
    prediction=Dense(1, activation='linear')(x)
    model=Model(inputs=inputs, outputs=prediction)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
        
#cols=['parcelid',
# 'airconditioningtypeid',
# 'architecturalstyletypeid',
# 'basementsqft',
# 'bathroomcnt',
# 'bedroomcnt',
# 'buildingclasstypeid',
# 'buildingqualitytypeid',
# 'calculatedbathnbr',
# 'decktypeid',
# 'finishedfloor1squarefeet',
# 'calculatedfinishedsquarefeet',
# 'finishedsquarefeet12',
# 'finishedsquarefeet13',
# 'finishedsquarefeet15',
# 'finishedsquarefeet50',
# 'finishedsquarefeet6',
# 'fips',
# 'fireplacecnt',
# 'fullbathcnt',
# 'garagecarcnt',
# 'garagetotalsqft',
# 'hashottuborspa',
# 'heatingorsystemtypeid',
# 'latitude',
# 'longitude',
# 'lotsizesquarefeet',
# 'poolcnt',
# 'poolsizesum',
# 'pooltypeid10',
# 'pooltypeid2',
# 'pooltypeid7',
# 'propertycountylandusecode',
# 'propertylandusetypeid',
# 'propertyzoningdesc',
# 'rawcensustractandblock',
# 'regionidcity',
# 'regionidcounty',
# 'regionidneighborhood',
# 'regionidzip',
# 'roomcnt',
# 'storytypeid',
# 'threequarterbathnbr',
# 'typeconstructiontypeid',
# 'unitcnt',
# 'yardbuildingsqft17',
# 'yardbuildingsqft26',
# 'yearbuilt',
# 'numberofstories',
# 'fireplaceflag',
# 'structuretaxvaluedollarcnt',
# 'taxvaluedollarcnt',
# 'assessmentyear',
# 'landtaxvaluedollarcnt',
# 'taxamount',
# 'taxdelinquencyflag',
# 'taxdelinquencyyear',
# 'censustractandblock']
#
#GB_params={'learning_rate':[0.1, 0.2, 0.5], 'n_estimators':[50, 100, 150, 200], 
#           'max_depth':[2, 3, 5, 10]}
log_errors=pd.read_csv("~/Documents/Machine Learning/Zillow/train_2016_v2.csv", low_memory=False)
#Data=pd.DataFrame(columns=cols)
#for df in pd.read_csv("~/Documents/Machine Learning/Zillow/properties_2016.csv",
#                 low_memory=False, chunksize=10**5):
#    labeled_data_in_chunk=df.merge(log_errors, on='parcelid', how='inner')
#    Data=pd.concat([Data, labeled_data_in_chunk])
#del Data['logerror']
#del Data['transactiondate']    
#Predictions=pd.DataFrame(columns=['parcelid', 'prediction', 'month', 'year'])
#start_time=time.time()
#
#HT_mask=Data.hashottuborspa =='nan'
#Data.loc[HT_mask,'hashottuborspa']=0
#Data.loc[~HT_mask,'hashottuborspa']=1
#
#FP_mask=Data.fireplaceflag == 'nan'
#Data.loc[FP_mask, 'fireplaceflag']=0
#Data.loc[~FP_mask, 'fireplaceflag']=1
#
#TD_mask=Data.taxdelinquencyflag == 'nan'
#Data.loc[TD_mask, 'taxdelinquencyflag']=0
#Data.loc[~TD_mask, 'taxdelinquencyflag']=1
#ls=list(Data)
#GB_params={'learning_rate':[0.1, 0.2, 0.5], 'n_estimators':[50, 100, 150, 200], 
#       'max_depth':[2, 3, 5, 10]}
#for to_predict in ls:
#    msk=Data[to_predict].notnull()
#    Non_null_data=Data[msk].copy()
#    if (len(Non_null_data)>=10) and (to_predict!='propertycountylandusecode'):
#        Non_null_data=Non_null_data.apply(pd.to_numeric, args=('coerce',)).fillna(value=0)
#        if len(Non_null_data)<len(Data) and len(Non_null_data)>100 :
#            int_msk=np.random.rand(len(Non_null_data))<0.4
#            train_y=Non_null_data[to_predict][int_msk].copy()
#            test_y=Non_null_data[to_predict][~int_msk].copy()
#            del Non_null_data[to_predict]
#            a=StandardScaler().fit(Non_null_data)
#            Non_null_data=pd.DataFrame(a.transform(Non_null_data), columns=list(Non_null_data))
#            int_train=Non_null_data[int_msk]
#            int_test=Non_null_data[~int_msk]
#            GB_CV=GridSearchCV(GradientBoostingRegressor(), GB_params).fit(int_train, train_y)
#            score=GB_CV.score(int_test, test_y)
#            print(to_predict+" "+str(score))
#            if score>=0:
#                Null_data=Data[~msk].copy()
#                del Null_data[to_predict]
#                Null_data=Null_data.apply(pd.to_numeric, args=('coerce',)).fillna(value=0)
#                Null_data=Null_data.fillna(value=0)
#                Null_data=pd.DataFrame(a.transform(Null_data), columns=list(Null_data))
#                Imputations=GB_CV.predict(Null_data)
#                Data.loc[~msk, to_predict]=Imputations
#Data=Data.apply(pd.to_numeric, args=('coerce',)).fillna(value=0)
Data=pd.read_csv("~/Documents/Machine Learning/Zillow/InSamplePropertiesNoResponse.csv")
labeled_data=pd.merge(Data, log_errors, how='inner', on='parcelid')
transaction_dates=labeled_data['transactiondate']
labeled_data['transactiondate']=pd.to_datetime(labeled_data['transactiondate'])
labeled_data['month']=labeled_data['transactiondate'].apply(lambda x: x.month)
labeled_data['year']=labeled_data['transactiondate'].apply(lambda x: x.year)
del labeled_data['transactiondate']
y_values=labeled_data['logerror']
del labeled_data['logerror']
data_dict=unsupervised_preprocess(labeled_data)
result_dict={}
NN_scores=[]
for n in range(100):
    msk=np.random.rand(len(labeled_data))<0.5
    train_dict={}
    test_dict={}
    train_y=y_values[msk]
    val_msk=np.random.rand(len(train_y.copy()))<.8
    validation_y=train_y[~val_msk]
    train_y=train_y[val_msk]
    test_y=y_values[~msk]
   
    for key, val in data_dict.iteritems():
        train_dict[key]=val[msk]
        test_dict[key]=val[~msk]
    validation=train_dict['Whitened'][~val_msk]
    train_dict['Whitened']=train_dict['Whitened'][val_msk]
    for k in range(1):
        p=np.exp(np.random.uniform(-3, -1))
        skip_length=np.random.randint(low=3, high=6)
        depth=np.random.randint(low=5, high=50)
        width=np.random.randint(low=20, high=500)
        pat=np.random.randint(low=100, high=250)
        model=functional_regression_model(prob=p, input_dim=len(train_dict['Whitened'].columns),
                                         skip_length=skip_length, width=width, depth=depth)
        early_stopping=EarlyStopping(monitor='val_loss', min_delta=1, patience=pat)
        hist=model.fit(np.asmatrix(train_dict['Whitened']), np.asmatrix(train_y).transpose(),
                  validation_data=(np.asmatrix(validation), np.asmatrix(validation_y).transpose()),
                  callbacks=[early_stopping], epochs=5000, batch_size=10000, verbose=0)
        score=np.mean(np.abs(model.predict(np.asmatrix(test_dict['Whitened']))[0]-test_y))
        print(str(score))
        NN_scores.append(score)
#    for key in data_dict:
#        print(key)
#        train_data=train_dict[key]
#        test_data=test_dict[key]
#        Production_Lasso=LassoCV().fit(train_data, train_y)
#        Lassoscore=np.mean(np.abs(Production_Lasso.predict(test_data)-test_y))
#        print("Lasso Validation MAE is "+str(Lassoscore))
#        Production_Ridge=RidgeCV().fit(train_data, train_y)
#        Ridgescore=np.mean(np.abs(Production_Ridge.predict(test_data)-test_y))
#        print("Ridge Validation MAE is "+str(Ridgescore))
#        Production_GB=GridSearchCV(GradientBoostingRegressor(), GB_params).fit(train_data, train_y)
#        GBscore=np.mean(np.abs(Production_GB.predict(test_data)-test_y))
#        print("GBR Validation MAE is "+str(GBscore))
#        result_dict[key+"{0}".format(n)]=np.minimum.reduce([Lassoscore, Ridgescore, GBscore])
        ##Maybe am aberration, but Factor Analysis n=30, GBR, gives me .0634


