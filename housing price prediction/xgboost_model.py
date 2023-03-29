#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 11:38:13 2022

@author: emmanuel
"""



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import pandas as pd
from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.metrics import mean_squared_error as mse


train = pd.read_csv('/home/emmanuel/Desktop/kaggle/home-data-for-ml-course/train.csv')
test = pd.read_csv('/home/emmanuel/Desktop/kaggle/home-data-for-ml-course/test.csv')



# YrSold was eliminated because the year will never come again, but the effect or the month may reflect a seasonal demand
features = ['MSSubClass','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
            'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
            'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',
            'ScreenPorch','PoolArea','MiscVal','MoSold','SalePrice']

features_X = ['MSSubClass','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
            'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
            'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',
            'ScreenPorch','PoolArea','MiscVal','MoSold']

clean_train = train[features] 
clean_test = test[features_X]

# # remove nan values
# clean_train = clean_train.dropna(axis=0)
# clean_test = clean_test.dropna(axis=0)

# replace nan values with 0 since these columns are numerical values
clean_train = clean_train.fillna(0)
clean_test = clean_test.fillna(0)

train_X = clean_train[features_X]
train_y = clean_train['SalePrice']

test_X = clean_test[features_X]
#test_y = clean_test['SalePrice']



DT_model = DT(max_leaf_nodes = 2000, random_state = 1)
DT_model.fit(train_X,train_y)

RF_model = RF(n_estimators = 100, random_state = 0)
RF_model.fit(train_X,train_y)

RF_prediction = RF_model.predict(train_X)
DT_prediction = DT_model.predict(train_X)
print(test_X.shape[0])
print('Decision tree error is ', mse(DT_prediction, train_y), ' where as Random forest error is ',mse(RF_prediction, train_y) )



# Run the code to save predictions in the format used for competition scoring
predict_y = DT_model.predict(test_X)
print(predict_y.shape[0])
#help(pd.DataFrame())
output = pd.DataFrame({'Id': test.Id, 'SalePrice': predict_y})
output.to_csv('Emmanuel_submit_house_price_prediction.csv', index=False)