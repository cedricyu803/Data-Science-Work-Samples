# -*- coding: utf-8 -*-
"""
Created on Sun Aug 01 11:00:00 2021

@author: Cedric Yu
"""

# import matplotlib.pyplot as plt
# import math
# import re


"""
#####################################

# https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview

# This challenge serves as final project for the "How to win a data science competition" Coursera course.

# In this competition you will work with a challenging time-series dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms - 1C Company. 

# We are asking you to predict total sales (count) <<for every product and store>> in the next <<month>>. By solving this competition you will be able to apply and enhance your data science skills.

#####################################

#!!! Submissions are evaluated by <<root mean squared error (RMSE)>>. 
#!!! <<True target values are clipped into [0,20] range>>

# Submission File

#!!! For each id in the test set, you must predict a <<total number of sales>>. The file should contain a header and have the following format:

# ID,item_cnt_month
# 0,0.5
# 1,0.5
# 2,0.5
# 3,0.5
# etc.

#####################################

# File descriptions
# sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
# test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.
# sample_submission.csv - a sample submission file in the correct format.
# items.csv - supplemental information about the items/products.
# item_categories.csv  - supplemental information about the items categories.
# shops.csv- supplemental information about the shops.

#####################################

# Data fields
# ID - an Id that represents a (Shop, Item) tuple within the <test set>
# shop_id - unique identifier of a shop
# item_id - unique identifier of a product
# item_category_id - unique identifier of item category
# item_cnt_day - number of products sold. You are predicting a monthly amount of this measure
# item_price - current price of an item
# date - date in format dd/mm/yyyy
# date_block_num - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33
# item_name - name of item
# shop_name - name of shop
# item_category_name - name of item category

# This dataset is permitted to be used for any purpose, including commercial use.



"""

#%% Preamble

import pandas as pd
# Make the output look better
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 75)
pd.options.mode.chained_assignment = None  # default='warn' # ignores warning about dropping columns inplace
import numpy as np
import matplotlib.pyplot as plt
import re


"""

baseline trial: use only the given features in sales_train.csv without any feature engineering

"""

#%% load dataset train.csv

# import original training dataset
train_df_raw = pd.read_csv(r'datasets\sales_train.csv', low_memory=False)
# train_df_raw.reset_index(inplace=True)
# train_df_raw.drop('index', inplace=True, axis=1)
train_df_raw.shape
# train_df_raw.shape
# (2935849, 6)

# train_df_head = train_df_raw.head(10000)

train_df_raw.describe()
#        date_block_num       shop_id       item_id    item_price  \
# count    2.935849e+06  2.935849e+06  2.935849e+06  2.935849e+06   
# mean     1.456991e+01  3.300173e+01  1.019723e+04  8.908532e+02   
# std      9.422988e+00  1.622697e+01  6.324297e+03  1.729800e+03   
# min      0.000000e+00  0.000000e+00  0.000000e+00 -1.000000e+00   
# 25%      7.000000e+00  2.200000e+01  4.476000e+03  2.490000e+02   
# 50%      1.400000e+01  3.100000e+01  9.343000e+03  3.990000e+02   
# 75%      2.300000e+01  4.700000e+01  1.568400e+04  9.990000e+02   
# max      3.300000e+01  5.900000e+01  2.216900e+04  3.079800e+05   

#        item_cnt_day  
# count  2.935849e+06  
# mean   1.242641e+00  
# std    2.618834e+00  
# min   -2.200000e+01  
# 25%    1.000000e+00  
# 50%    1.000000e+00  
# 75%    1.000000e+00  
# max    2.169000e+03  

# train_df_raw['item_cnt_day'].nlargest(3)

# 2909818    2169.0
# 2326930    1000.0
# 2864235     669.0
# Name: item_cnt_day, dtype: float64

#%% drop outliers

quantile = 1e-5

train_df = train_df_raw[ 
    (train_df_raw['item_cnt_day'] > train_df_raw['item_cnt_day'].quantile(quantile)) 
    & (train_df_raw['item_cnt_day'] < train_df_raw['item_cnt_day'].quantile(1 - quantile)) 
    & (train_df_raw['item_price'] > train_df_raw['item_price'].quantile(quantile)) 
    & (train_df_raw['item_price'] < train_df_raw['item_price'].quantile(1 - quantile)
       )]

# train_df_raw.shape # (2935849, 6)
# train_df.shape # (2932744, 6)


#%% convert 'date' to datetime

from datetime import datetime
import time

def date_datetime(row):
    row['date'] = time.mktime(datetime.strptime(row['date'], "%d.%m.%Y").timetuple() )
    return row
# 'date' columm dtype: datetime64[ns]
train_df = train_df.apply(date_datetime, axis='columns')


#%% train-validation split

month_split = 29

Xy_train = train_df[ train_df['date_block_num'] < month_split ]
X_train = Xy_train.drop(['item_cnt_day', 'date'], axis = 1)
y_train = Xy_train['item_cnt_day']

Xy_valid = train_df[ train_df['date_block_num'] >= month_split ]
X_valid = Xy_valid.drop(['item_cnt_day', 'date'], axis = 1)
y_valid = Xy_valid['item_cnt_day']



#%% scaler

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(X_train)
encoded_X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns, index = X_train.index)
encoded_X_valid_scaled = pd.DataFrame(scaler.transform(X_valid), columns = X_valid.columns, index = X_valid.index)


#%% model scores

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

linreg = LinearRegression(n_jobs = -1)
linreg.fit(encoded_X_train_scaled, y_train)

print('Linear Regression: training set R2 score is {:.3f}'.format(linreg.score(encoded_X_train_scaled, y_train)))
print('Linear Regression: validation set R2 score is {:.3f}'.format(linreg.score(encoded_X_valid_scaled, y_valid)))
# Linear Regression: training set R2 score is 0.030
# Linear Regression: validation set R2 score is 0.051

print('Linear Regression: training set root-mean-squared error is {:.0f}'.format(mean_squared_error(y_train, linreg.predict(encoded_X_train_scaled), squared = False)))
print('Linear Regression: validation set root-mean-squared error is {:.0f}'.format(mean_squared_error(y_valid, linreg.predict(encoded_X_valid_scaled), squared = False)))
# Linear Regression: training set root-mean-squared error is 2
# Linear Regression: validation set root-mean-squared error is 2


plt.figure()
plt.scatter(np.array(list(y_train.index)), np.array(list(y_train.values)))
plt.scatter(np.array(list(y_train.index)), linreg.predict(X_train))

plt.figure()
plt.scatter(np.array(list(y_valid.index)), np.array(list(y_valid.values)))
plt.scatter(np.array(list(y_valid.index)), linreg.predict(X_valid))

from xgboost import XGBRegressor

XGBR_model = XGBRegressor(eval_metric = "rmse")
XGBR_model.fit(encoded_X_train_scaled, y_train)
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, eval_metric='rmse',
#              gamma=0, gpu_id=-1, importance_type='gain',
#              interaction_constraints='', learning_rate=0.300000012,
#              max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan,
#              monotone_constraints='()', n_estimators=100, n_jobs=8,
#              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
#              scale_pos_weight=1, subsample=1, tree_method='exact',
#              validate_parameters=1, verbosity=None)

print(mean_squared_error(y_train, XGBR_model.predict(encoded_X_train_scaled), squared = False))
print(mean_squared_error(y_valid, XGBR_model.predict(encoded_X_valid_scaled), squared = False))
# 1.3167275901365347
# 1.821634266195159


XGBR_feature_importances = pd.Series(XGBR_model.feature_importances_, index = X_train.columns)
# date_block_num    0.148278
# shop_id           0.328945
# item_id           0.157161
# item_price        0.365616
# dtype: float32

plt.figure()
plt.scatter(np.array(list(y_train.index)), np.array(list(y_train.values)), label = 'training set data')
plt.scatter(np.array(list(y_train.index)), XGBR_model.predict(encoded_X_train_scaled), label = 'model prediction')
plt.legend()
plt.title('XGBoost default, kept all given features with minimal feature engineering')
plt.savefig('XGBoost default, kept all given features with minimal feature engineering.png', dpi = 600)

plt.figure()
plt.scatter(np.array(list(y_valid.index)), np.array(list(y_valid.values)), label = 'validation set data')
plt.scatter(np.array(list(y_valid.index)), XGBR_model.predict(encoded_X_valid_scaled), label = 'model prediction')
plt.legend()
plt.title('XGBoost default, kept all given features with minimal feature engineering')
plt.savefig('XGBoost default, kept all given features with minimal feature engineering2.png', dpi = 600)



#%% groupby month

""" # Next, we group by month and drop 'date' column, extract month and year from date_block_num and drop date_block_num """
""" # rmse increases dramatically """


train_df_month = train_df_raw.copy()

# drop outliers

quantile = 1e-5

train_df_month = train_df_raw[ 
    (train_df_raw['item_cnt_day'] > train_df_raw['item_cnt_day'].quantile(quantile)) 
    & (train_df_raw['item_cnt_day'] < train_df_raw['item_cnt_day'].quantile(1 - quantile)) 
    & (train_df_raw['item_price'] > train_df_raw['item_price'].quantile(quantile)) 
    & (train_df_raw['item_price'] < train_df_raw['item_price'].quantile(1 - quantile)
       )]

# train_df_raw.shape # (2935849, 6)
# train_df_month.shape # (2932744, 6)


train_df_month = train_df_month.drop(['date'], axis = 1)
train_df_month = train_df_month.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_price': np.nanmean, 'item_cnt_day': np.nansum})
train_df_month.reset_index(inplace = True)
train_df_month.rename(columns = {'item_price': 'item_price_mean', 'item_cnt_day': 'item_cnt_month'}, inplace = True)

train_df_month['month_of_year'] = train_df_month['date_block_num'] % 12 + 1
train_df_month['year'] = train_df_month['date_block_num'] // 12 + 1

#%% convert 'date' to datetime

# from datetime import datetime
# import time

# def date_datetime(row):
#     row['date'] = time.mktime(datetime.strptime(row['date'], "%d.%m.%Y").timetuple() )
#     return row
# # 'date' columm dtype: datetime64[ns]
# train_df = train_df.apply(date_datetime, axis='columns')


#%% train-validation split

month_split = 29

Xy_train2 = train_df_month[ train_df_month['date_block_num'] < month_split ]
X_train2 = Xy_train2.drop(['item_cnt_month', 'date_block_num'], axis = 1)
y_train2 = Xy_train2['item_cnt_month']

Xy_valid2 = train_df_month[ train_df_month['date_block_num'] >= month_split ]
X_valid2 = Xy_valid2.drop(['item_cnt_month', 'date_block_num'], axis = 1)
y_valid2 = Xy_valid2['item_cnt_month']

#%% scaler

from sklearn.preprocessing import MinMaxScaler
scaler2 = MinMaxScaler()

scaler2.fit(X_train2)
encoded_X_train_scaled2 = pd.DataFrame(scaler2.transform(X_train2), columns = X_train2.columns, index = X_train2.index)
encoded_X_valid_scaled2 = pd.DataFrame(scaler2.transform(X_valid2), columns = X_valid2.columns, index = X_valid2.index)

from sklearn.linear_model import LinearRegression

linreg2 = LinearRegression(n_jobs = -1)
linreg2.fit(encoded_X_train_scaled2, y_train2)


print('Linear Regression: training set root-mean-squared error is {:.0f}'.format(mean_squared_error(y_train2, linreg2.predict(encoded_X_train_scaled2), squared = False)))
print('Linear Regression: validation set root-mean-squared error is {:.0f}'.format(mean_squared_error(y_valid2, linreg2.predict(encoded_X_valid_scaled2), squared = False)))
# Linear Regression: training set root-mean-squared error is 8
# Linear Regression: validation set root-mean-squared error is 8


XGBR_model2 = XGBRegressor(eval_metric = "rmse")
XGBR_model2.fit(encoded_X_train_scaled2, y_train2)
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, eval_metric='rmse',
#              gamma=0, gpu_id=-1, importance_type='gain',
#              interaction_constraints='', learning_rate=0.300000012,
#              max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan,
#              monotone_constraints='()', n_estimators=100, n_jobs=8,
#              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
#              scale_pos_weight=1, subsample=1, tree_method='exact',
#              validate_parameters=1, verbosity=None)

print(mean_squared_error(y_train2, XGBR_model2.predict(encoded_X_train_scaled2), squared = False))
print(mean_squared_error(y_valid2, XGBR_model2.predict(encoded_X_valid_scaled2), squared = False))
# 3.8620054926653404
# 6.566801662625399

XGBR_feature_importances2 = pd.Series(XGBR_model2.feature_importances_, index = X_train2.columns)

# date_block_num     0.147090
# shop_id            0.393532
# item_id            0.110679
# item_price_mean    0.348698
# dtype: float32

from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor()
GBR.fit(X_train2, y_train2)
print(mean_squared_error(y_train2, GBR.predict(X_train2), squared = False))
print(mean_squared_error(y_valid2, GBR.predict(X_valid2), squared = False))
# 6.964224867868832
# 7.266205394634348



#%%

test_df_all = pd.read_csv(r'datasets\test.csv')
# (214200, 3)


X_pred = pp.drop_cols(test_df_all)
OH_X_pred = pp.not_train_preprocess(X_pred)

clf_gb.predict(OH_X_pred)
# array([0., 0., 0., ..., 0., 0., 1.])

y_test_predict_proba = pd.Series(clf_gb.predict_proba(OH_X_pred)[:,1], index= test_df_all['ticket_id'], name='compliance')







