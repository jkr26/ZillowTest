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
Data=pd.read_csv("~/Documents/Machine Learning/Zillow/Train_1_filled_in.csv", low_memory=False)
Data['logerror']=pd.read_csv("~/Documents/Machine Learning/Zillow/Train_1.csv", low_memory=False)['logerror']
Data_filled_na=Data#.fillna(value=0)

to_del=['Unnamed: 0']

for string in to_del:
    del Data_filled_na[string]



ls=list(Data_filled_na)


GB_params={'learning_rate':[0.1, 0.2, 0.5], 'n_estimators':[50, 100, 150, 200], 
           'max_depth':[2, 3, 5, 10]}


##Well filling in the missing data worked v. well.

##There are 2 clear paths forward--bang you NN the problem
##or you preprocess further, e.g. PCA or feature selection or both, then you NN the problem.

##This approach alone yields an R^2 of .587 and reduced test MAE from .0688 to .0194!
        
        
Data_filled_na=Data_filled_na.astype(str).convert_objects(convert_numeric=True)
Response=Data_filled_na['logerror'].copy()
del Data_filled_na['logerror']
Data_filled_na=Data_filled_na.fillna(value=0)
Normalized_data=StandardScaler().fit_transform(Data_filled_na)
msk=np.random.rand(len(Data_filled_na))
#train=Normalized_data[msk<0.8]
#test=Normalized_data[msk>=0.8]
train=Data_filled_na[msk<0.7]
test=Data_filled_na[msk>=0.7]

train_y=Response[msk<0.7]
test_y=Response[msk>=0.7]

#Bucket_response=

GB_params={'loss': ['ls', 'lad', 'huber', 'quantile'], 'learning_rate':[0.1, 0.2, 0.5], 'n_estimators':[50, 100, 150, 200], 
           'max_depth':[2, 3, 5, 10]}
gb_reg=GridSearchCV(GradientBoostingRegressor(), GB_params).fit(train, train_y)

gb_score=gb_reg.score(test, test_y)
print("--- %s seconds---" %(time.time()-start_time))