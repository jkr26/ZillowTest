#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from keras import models
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical


"""
Created on Mon Jun 12 19:26:14 2017

@author: jkr
"""
##Some slight preprocessing
Data=pd.read_csv("~/Documents/Machine Learning/Zillow/Train_1_filled_in.csv")
Data['logerror']=pd.read_csv("~/Documents/Machine Learning/Zillow/Train_1.csv")['logerror']

del Data['Unnamed: 0']

large_error_msk=abs(Data['logerror'])<1
Real_data=Data[large_error_msk].copy()

bins=[-0.9943 , -0.79488, -0.59546, -0.39604, -0.19662,  0.0028 ,
         0.20222,  0.40164,  0.60106,  0.80048,  0.9999 ]
         
digitized=np.digitize(Real_data['logerror'], bins)

Real_data['Digitized_logerror']=digitized

del Real_data['logerror']

msk=np.random.rand(len(Real_data))<0.5
train=Real_data[msk]
test=Real_data[~msk]

train_y=train['Digitized_logerror']
test_y=test['Digitized_logerror']

train_y=to_categorical(train_y)
test_y=to_categorical(test_y)

del train['Digitized_logerror']
del test['Digitized_logerror']



##Simple neural net construction, just experimentation, NOT real final model!
model=models.Sequential()
model.add(Dense(units=100, input_dim=54))
model.add(Activation('relu'))
#model.add(Dense(units=200))
#model.add(Activation('relu'))
#model.add(Dense(units=500))
#model.add(Activation('relu'))
#model.add(Dense(units=500))
#model.add(Activation('relu'))
#model.add(Dense(units=500))
#model.add(Activation('relu'))
#model.add(Dense(units=500))
#model.add(Activation('relu'))
#model.add(Dense(units=500))
#model.add(Activation('relu'))
#model.add(Dense(units=500))
#model.add(Activation('relu'))
#model.add(Dense(units=500))
model.add(Dense(100, activation='relu'))
model.add(Dense(len(train_y[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(np.array(train), np.array(train_y), epochs=100, batch_size=100)
