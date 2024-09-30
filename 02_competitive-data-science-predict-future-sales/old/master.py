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

#%% Workflow

"""

# load training (train + validation) set and test set
# drop outliers: keep item_cnt_day in (0, 750), and item_price in (0.1, 100000)
# drop date column, and add 'month' and 'year' columns to both train + validation and test sets
# compute 'item_cnt_month' and 'item_price_mean' for each month
# concatenate train + validation and test sets into a single df
# append shop_city, shop_type, 'item_category_id', 'item_main_category', 'platform' columns
# create 4-month lag features for lag_cols = ['item_cnt_month', 'item_price_mean']

# we choose a time-ordered split of train and validation sets, and the latter to span the last three months (Aug - Oct 2015)

feature engineering: 

# train:

# merge duplicate shop pairs
# only keep the shops that are still open (open for at least 3 months, open last month) and add column of the fraction of which month they first opened)
# only keep items that are still sold (sold for at least 3 months, and sold in last 3 months), add a column of the fraction of which month they were first sold if sold at all
# drop(['date_block_num']
# select feature columns: feature_cols = ['month', 'year', 'shop_city', 'shop_type', 'item_main_category',
#        'platform', 'first_open_month_num'] + [col + '_lag_' + str(i+1) for i in range(lag_months) for col in lag_cols]
# frequency-encode categorical features
# drop 'platform_encoded' column (it is not useful)
# min-max scaling

# valid and test:

# merge duplicate shop pairs
# X_valid = pp.shop_duplicates(X_valid)
# add 'first_open_month_num' for each shop; if it is a new shop, set it to one (i.e. assume it first opened at the end of training period)
# drop(['date_block_num']
# select feature columns
# frequency-encode categorical features
# drop 'platform_encoded' column (it is not useful)
# min-max scaling


# clip the target values in train and validation to [0., 20.]

# model training using validation rmse: 
# random forest regressor with random CV search
# XGboost regressor (default hyperparameters)
# save models to external files using pickle


# clip test prediction to [0., 20.]

"""




#%% Preamble

import pandas as pd
# Make the output look better
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None  # default='warn' # ignores warning about dropping columns inplace
import numpy as np
import matplotlib.pyplot as plt
# import re

import re
import os
os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\4_competitive-data-science-predict-future-sales')



#%% load dataset train.csv

train_df_raw_cols = pd.read_csv(r'datasets\sales_train.csv', nrows = 0)
train_df_raw_cols = train_df_raw_cols.drop(['date'], axis = 1)
train_df_raw_cols = train_df_raw_cols.columns.to_list()

# import original training dataset
train_df_raw = pd.read_csv('datasets/sales_train.csv', low_memory=False, usecols = train_df_raw_cols)
# train_df_raw.shape
# (2935849, 5)

#%% load test set
test_df_all = pd.read_csv(r'datasets\test.csv')
# (214200, 3)
test_df = test_df_all.copy()
test_df_id = test_df_all.copy()

#%% import preprocessing functions
import preprocessing_util as pp

#%% drop outliers

train_df_month = train_df_raw.copy()

train_df_month = train_df_month[ 
    (train_df_month['item_cnt_day'] > 0) 
    & (train_df_month['item_cnt_day'] < 750) 
    & (train_df_month['item_price'] > 0.01) 
    & (train_df_month['item_price'] < 75000
       )]

# train_df_raw.shape # (2935849, 5)

#%% add 'month' and 'year' columns, aggregate monthy

"""aggregate monthly for each ('shop_id', 'item_id') """

# train_df_month = train_df_month.drop(['date'], axis = 1)
train_df_month = train_df_month.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_price': np.nanmean, 'item_cnt_day': np.nansum})
train_df_month.reset_index(inplace = True)
train_df_month.rename(columns = {'item_price': 'item_price_mean', 'item_cnt_day': 'item_cnt_month'}, inplace = True)


""" add 'month', 'year' columns"""
train_df_month['month'] = train_df_month['date_block_num'] % 12 + 1
train_df_month['year'] = train_df_month['date_block_num'] // 12 + 1

"""# sort by date_block_num"""
train_df_month = train_df_month.sort_values(['year', 'month'], ascending = True)

""" assign datetime dolumns to test set"""
test_df['date_block_num'] = 34
test_df['year'] = 3
test_df['month'] = 11

""" concat training and test sets"""
train_df_month_test = pd.concat([train_df_month, test_df], axis = 0)

#%% append shop columns and item columns

"""append shop columns and item columns"""
train_df_month_test = pd.merge(train_df_month_test, shop_id_city_type, how = 'left', on = 'shop_id')
train_df_month_test = pd.merge(train_df_month_test, items_id_cat_platform, how = 'left', on = 'item_id')

# train_df_month.shape
# (1608224, 7)


#%% lag (not advanced) features from target

"""# our dataset is of consecutive months--- we create lag features of a given column simply by shifting the date_block_num value by + 1 (so that it is a <lag> feature) """

def lags(df, col, n=3) :
    df0 = df.copy()
    
    for i in np.arange(1, n + 1) :
        df_lag = df0[['shop_id', 'item_id', 'date_block_num', col]].copy()
        df_lag['date_block_num'] = df_lag['date_block_num'] + 1
        df_lag.rename(columns = {col: col + '_lag_' + str(i)}, inplace = True)
        df0 = pd.merge(df0, df_lag, how = 'left', on = ['shop_id', 'item_id', 'date_block_num'])
        df0[col + '_lag_' + str(i)].fillna(0., inplace = True)
    
    return df0


#%% create lag features from target

lag_months = 14
lag_cols = ['item_cnt_month', 'item_price_mean']

for col in lag_cols : 
    train_df_month_test = lags(train_df_month_test, col, lag_months)

""" keep lag features: 1-3, 12-14 months for item_cnt_month, and 1-2, 11-12 months for item_price_mean"""
train_df_month_test = train_df_month_test.drop(['item_cnt_month_lag_' + str(i) for i in np.arange(4, 12)], axis = 1)
train_df_month_test = train_df_month_test.drop(['item_price_mean_lag_' + str(i) for i in np.arange(3, 11)], axis = 1)
train_df_month_test = train_df_month_test.drop(['item_price_mean_lag_' + str(i) for i in np.arange(13, 15)], axis = 1)


#%% train-validation split

""" We used largest lag month = 14.  """

# we do a time series split; training set contains March 2014 to Jul 2015, validation set contains Aug 2015 - Oct 2015

df_train = train_df_month_test[(train_df_month_test['date_block_num'] > lag_months - 1) & (train_df_month_test['date_block_num'] < 31 )]

df_valid = train_df_month_test[(train_df_month_test['date_block_num'] > 30) & (train_df_month_test['date_block_num'] < 34 )]

df_test = train_df_month_test[ train_df_month_test['date_block_num'] == 34 ]

#%% feature-target split

X_train = df_train.drop(['item_cnt_month', 'ID'], axis = 1)
y_train = df_train['item_cnt_month']
# print(list(X_train.columns))
# ['date_block_num', 'shop_id', 'item_id', 'item_price_mean', 'month', 'year', 'shop_city', 'shop_type', 'item_category_id', 'item_main_category', 'platform', 'item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3', 'item_price_mean_lag_1', 'item_price_mean_lag_2', 'item_price_mean_lag_3']

# X_train0 = X_train.copy()
# y_train0 = y_train.copy()

X_valid = df_valid.drop(['item_cnt_month', 'item_price_mean', 'ID'], axis = 1)
y_valid = df_valid['item_cnt_month']

# X_valid0 = X_valid.copy()
# y_valid0 = y_valid.copy()

X_test = df_test.drop(['item_cnt_month', 'item_price_mean', 'ID'], axis = 1)
y_test = df_test[['ID', 'shop_id', 'item_id', 'item_cnt_month']]

#%% execute the above fits and transforms


good_shops_items = pp.good_shops_items0()
# train_item_price_mean = train_item_price_mean0()
# item_cnt_year_avg = pp.item_cnt_year_avg0()
freq_encoder = pp.FrequencyEncoder()
freq_cols = ['shop_city', 'shop_type', 'item_main_category', 'platform']
feature_cols = ['month', 'year', 'shop_city', 'shop_type', 'item_main_category',
       'platform', 'first_open_month_num'] + [col + '_lag_' + str(i+1) for i in range(lag_months) for col in lag_cols]
# , 'item_cnt_month_by_shop', 'item_cnt_month_by_item'
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train = pp.shop_duplicates(X_train)
X_train, y_train = good_shops_items.fit(X_train, y_train)
# train_item_price_mean.fit(X_train)
# item_cnt_year_avg.fit(X_train, y_train)
# X_train = item_cnt_year_avg.transform(X_train)
X_train.drop(['date_block_num'], axis = 1, inplace = True)
X_train = X_train[feature_cols]
freq_encoder.fit(X_train, freq_cols)
encoded_X_train = freq_encoder.transform(X_train)
encoded_X_train = encoded_X_train.drop(['platform_encoded'], axis = 1)
scaler.fit(encoded_X_train)
encoded_X_train_scaled = pd.DataFrame(scaler.transform(encoded_X_train), columns = encoded_X_train.columns, index = encoded_X_train.index)


X_valid = pp.shop_duplicates(X_valid)
X_valid = good_shops_items.transform(X_valid)
# X_valid = train_item_price_mean.transform(X_valid)
# X_valid = item_cnt_year_avg.transform(X_valid)
X_valid.drop(['date_block_num'], axis = 1, inplace = True)
X_valid = X_valid[feature_cols]
encoded_X_valid = freq_encoder.transform(X_valid)
encoded_X_valid = encoded_X_valid.drop(['platform_encoded'], axis = 1)
encoded_X_valid_scaled = pd.DataFrame(scaler.transform(encoded_X_valid), columns = encoded_X_valid.columns, index = encoded_X_valid.index)


X_test = pp.shop_duplicates(X_test)
X_test = good_shops_items.transform(X_test)
# X_test = train_item_price_mean.transform(X_test)
# X_test = item_cnt_year_avg.transform(X_test)
X_test.drop(['date_block_num'], axis = 1, inplace = True)
X_test = X_test[feature_cols]
encoded_X_test = freq_encoder.transform(X_test)
encoded_X_test = encoded_X_test.drop(['platform_encoded'], axis = 1)
encoded_X_test_scaled = pd.DataFrame(scaler.transform(encoded_X_test), columns = encoded_X_test.columns, index = encoded_X_test.index)


#%% clip item_cnt_month in the range [0, 20]

def clip_cnt(y) : 
    y = y.copy()
    y.clip(0., 20., inplace = True)
    return y

y_train = clip_cnt(y_train)
y_valid = clip_cnt(y_valid)
# train_df_month['item_cnt_month'].plot(kind = 'box')

#%% model scores

from sklearn.metrics import mean_squared_error

#################################

from sklearn.linear_model import LinearRegression

linreg = LinearRegression(n_jobs = -1)
linreg.fit(encoded_X_train_scaled, y_train)

print('Linear Regression: training set root-mean-squared error is {:.0f}'.format(mean_squared_error(y_train, linreg.predict(encoded_X_train_scaled), squared = False)))
print('Linear Regression: validation set root-mean-squared error is {:.0f}'.format(mean_squared_error(y_valid, linreg.predict(encoded_X_valid_scaled), squared = False)))
# Linear Regression: training set root-mean-squared error is 3
# Linear Regression: validation set root-mean-squared error is 2


plt.figure()
plt.scatter(np.array(list(y_train.index)), np.array(list(y_train.values)))
plt.scatter(np.array(list(y_train.index)), linreg.predict(encoded_X_train))

plt.figure()
plt.scatter(np.array(list(y_valid.index)), np.array(list(y_valid.values)))
plt.scatter(np.array(list(y_valid.index)), linreg.predict(encoded_X_valid))

#################################

# from sklearn.linear_model import Ridge

# alpha = 10.
# linridge = Ridge(alpha = alpha)
# linridge.fit(encoded_X_train_scaled, y_train)

# print('Ridge Regression with alpha = {}: training set root-mean-squared error is {:.3f}'.format(alpha, mean_squared_error(y_train, linridge.predict(encoded_X_train_scaled), squared = False)))
# print('Ridge Regression with alpha = {}: validation set root-mean-squared error is {:.3f}'.format(alpha, mean_squared_error(y_valid, linridge.predict(encoded_X_valid_scaled), squared = False)))



# linridge = Ridge()



# from sklearn.model_selection import GridSearchCV
# grid_values_ridge = {'alpha': list(np.arange(29, 80, 5))}

# linridge_mse = GridSearchCV(linridge, param_grid = grid_values_ridge, scoring = 'neg_root_mean_squared_error')
# linridge_mse.fit(encoded_X_train_scaled, y_train)
# print('Grid best parameter (min. neg_root_mean_squared_error): ', linridge_mse.best_params_)
# # Grid best parameter (min. neg_root_mean_squared_error):  {'alpha': 14}

# print('Ridge Regression with alpha = {}: training set R2 score is {:.3f}'.format(linridge_mse.best_params_['alpha'], linridge_mse.score(encoded_X_train_scaled, y_train)))
# print('Ridge Regression with alpha = {}: validation set R2 score is {:.3f}'.format(linridge_mse.best_params_['alpha'], linridge_mse.score(encoded_X_valid_scaled, y_valid)))
# # Ridge Regression with alpha = 14: training set R2 score is -21.229
# # Ridge Regression with alpha = 14: validation set R2 score is -40.024

# print('Ridge Regression with alpha = {}: training set root-mean-squared error is {:.0f}'.format(linridge_mse.best_params_['alpha'], mean_squared_error(y_train, linridge_mse.predict(encoded_X_train_scaled), squared = False)))
# print('Ridge Regression with alpha = {}: validation set root-mean-squared error is {:.0f}'.format(linridge_mse.best_params_['alpha'], mean_squared_error(y_valid, linridge_mse.predict(encoded_X_valid_scaled), squared = False)))
# # Ridge Regression with alpha = 14: training set root-mean-squared error is 21
# # Ridge Regression with alpha = 14: validation set root-mean-squared error is 40

# #################################

# from sklearn.linear_model import Lasso
# linlasso = Lasso(max_iter = 10000)

# from sklearn.model_selection import GridSearchCV
# grid_values_lasso = {'alpha': list(np.arange(0,1,0.5))}

# linlasso_mse = GridSearchCV(linlasso, param_grid = grid_values_lasso, scoring = 'neg_root_mean_squared_error')
# linlasso_mse.fit(encoded_X_train_scaled, y_train)
# print('Grid best parameter (min. neg_root_mean_squared_error): ', linlasso_mse.best_params_)
# # Grid best parameter (min. neg_root_mean_squared_error):  {'alpha': 68}

# print('Lasso Regression with alpha = {}: training set R2 score is {:.3f}'.format(linlasso_mse.best_params_['alpha'], linlasso_mse.score(encoded_X_train_scaled, y_train)))
# print('Lasso Regression with alpha = {}: validation set R2 score is {:.3f}'.format(linlasso_mse.best_params_['alpha'], linlasso_mse.score(encoded_X_valid_scaled, y_valid)))
# # Lasso Regression with alpha = 68: training set R2 score is -22411.582
# # Lasso Regression with alpha = 68: validation set R2 score is -46851.601

# print('Lasso Regression with alpha = {}: training set root-mean-squared error is {:.0f}'.format(linlasso_mse.best_params_['alpha'], mean_squared_error(y_train, linlasso_mse.predict(encoded_X_train_scaled), squared = False)))
# print('Lasso Regression with alpha = {}: validation set root-mean-squared error is {:.0f}'.format(linlasso_mse.best_params_['alpha'], mean_squared_error(y_valid, linlasso_mse.predict(encoded_X_valid_scaled), squared = False)))
# # Lasso Regression with alpha = 68: training set root-mean-squared error is 22412
# # Lasso Regression with alpha = 68: validation set root-mean-squared error is 46852

#################################

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3)
encoded_X_train_scaled_poly = poly.fit_transform(encoded_X_train_scaled)
encoded_X_valid_scaled_poly = poly.transform(encoded_X_valid_scaled)

from sklearn.linear_model import Ridge
linridge_poly = Ridge(alpha = 100.)
linridge_poly.fit(encoded_X_train_scaled_poly, y_train)
print('Ridge Polynomial (p = 3) Regression with alpha = {}: training set root-mean-squared log error is {:.0f}'.format(100., mean_squared_error(y_train, linridge_poly.predict(encoded_X_train_scaled_poly), squared = False)))
print('Ridge Polynomial (p = 3) Regression with alpha = {}: validation set root-mean-squared log error is {:.0f}'.format(100., mean_squared_error(y_valid, linridge_poly.predict(encoded_X_valid_scaled_poly), squared = False)))
# Ridge Polynomial (p = 3) Regression with alpha = 100.0: training set root-mean-squared log error is 2
# Ridge Polynomial (p = 3) Regression with alpha = 100.0: validation set root-mean-squared log error is 2




# grid_values_ridge_poly = {'alpha': list(np.arange(60,80,1))}

# linridge_mse_poly = GridSearchCV(linridge_poly, param_grid = grid_values_ridge_poly, scoring = 'neg_root_mean_squared_error')
# linridge_mse_poly.fit(encoded_X_train_scaled_poly, y_train)
# print('Grid best parameter (min. neg_root_mean_squared_error): ', linridge_mse_poly.best_params_)
# # Grid best parameter (min. neg_root_mean_squared_error):  {'alpha': 65}

# print('Ridge Polynomial (p = 2) Regression with alpha = {}: training set R2 score is {:.3f}'.format(linridge_mse_poly.best_params_['alpha'], linridge_mse_poly.score(encoded_X_train_scaled_poly, y_train)))
# print('Ridge Polynomial (p = 2) Regression with alpha = {}: validation set R2 score is {:.3f}'.format(linridge_mse_poly.best_params_['alpha'], linridge_mse_poly.score(encoded_X_valid_scaled_poly, y_valid)))
# # Ridge Polynomial (p = 2) Regression with alpha = 65: training set R2 score is -14758.921
# # Ridge Polynomial (p = 2) Regression with alpha = 65: validation set R2 score is -43564.468

# print('Ridge Polynomial (p = 2) Regression with alpha = {}: training set root-mean-squared log error is {:.0f}'.format(linridge_mse_poly.best_params_['alpha'], mean_squared_log_error(y_train, linridge_mse_poly.predict(encoded_X_train_scaled_poly), squared = False)))
# print('Ridge Polynomial (p = 2) Regression with alpha = {}: validation set root-mean-squared log error is {:.0f}'.format(linridge_mse_poly.best_params_['alpha'], mean_squared_log_error(y_valid, linridge_mse_poly.predict(encoded_X_valid_scaled_poly), squared = False)))
# # Ridge Polynomial (p = 2) Regression with alpha = 65: training set root-mean-squared error is 14759
# # Ridge Polynomial (p = 2) Regression with alpha = 65: validation set root-mean-squared error is 43564

#################################

# from sklearn.neighbors import KNeighborsRegressor
# k = 3
# knnreg = KNeighborsRegressor(n_neighbors = k).fit(encoded_X_train_scaled, y_train)

# print('knn Regression with k = {}: training set R2 score is {:.3f}'.format(k, knnreg.score(encoded_X_train_scaled, y_train)))
# print('knn Regression with k = {}: validation set R2 score is {:.3f}'.format(k, knnreg.score(encoded_X_valid_scaled, y_valid)))
# # knn Regression with k = 7: training set R2 score is 0.799
# # knn Regression with k = 7: validation set R2 score is 0.674

# print('knn Regression with k = {}: training set root-mean-squared log error is {:.0f}'.format(k, mean_squared_log_error(y_train, knnreg.predict(encoded_X_train_scaled), squared = False)))
# print('knn Regression with k = {}: validation set root-mean-squared log error is {:.0f}'.format(k, mean_squared_log_error(y_valid, knnreg.predict(encoded_X_valid_scaled), squared = False)))
# # knn Regression with k = 7: training set root-mean-squared error is 35158
# # knn Regression with k = 7: validation set root-mean-squared error is 47527

# knnreg2 = KNeighborsRegressor(n_neighbors = k)
# grid_values_knn = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7], 'p': [1, 2, 3]}

# knnreg2_mse = GridSearchCV(knnreg2, param_grid = grid_values_knn, scoring = 'neg_root_mean_squared_error')
# knnreg2_mse.fit(encoded_X_train_scaled, y_train)
# print('Grid best parameter (min. neg_root_mean_squared_error): ', knnreg2_mse.best_params_)
# # Grid best parameter (min. neg_root_mean_squared_error):  {'n_neighbors': 4, 'p': 1}

# print('knn Regression with k = {}, p = {}: training set R2 score is {:.3f}'.format(knnreg2_mse.best_params_['n_neighbors'], knnreg2_mse.best_params_['p'], knnreg2_mse.score(encoded_X_train_scaled, y_train)))
# print('knn Regression with k = {}, p = {}: validation set R2 score is {:.3f}'.format(knnreg2_mse.best_params_['n_neighbors'], knnreg2_mse.best_params_['p'], knnreg2_mse.score(encoded_X_valid_scaled, y_valid)))
# # knn Regression with k = 4, p = 1: training set R2 score is -29032.238
# # knn Regression with k = 4, p = 1: validation set R2 score is -44316.140

# print('knn Regression with k = {}, p = {}: training set root-mean-squared log error is {:.0f}'.format(knnreg2_mse.best_params_['n_neighbors'], knnreg2_mse.best_params_['p'], mean_squared_log_error(y_train, knnreg2_mse.predict(encoded_X_train_scaled), squared = False)))
# print('knn Regression with k = {}, p = {}: validation set root-mean-squared log error is {:.0f}'.format(knnreg2_mse.best_params_['n_neighbors'], knnreg2_mse.best_params_['p'], mean_squared_log_error(y_valid, knnreg2_mse.predict(encoded_X_valid_scaled), squared = False)))
# # knn Regression with k = 4, p = 1: training set root-mean-squared error is 29032
# # knn Regression with k = 4, p = 1: validation set root-mean-squared error is 44316

#################################


from xgboost import XGBRegressor

XGBR_model = XGBRegressor(eval_metric="rmse", n_jobs = 8)
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
# 2.044330190119043
# 1.866580220004559

XGBR_feature_importances = pd.Series(XGBR_model.feature_importances_, index = encoded_X_train_scaled.columns).sort_values(ascending = False)
# item_cnt_month_lag_1          0.634222
# item_main_category_encoded    0.113134
# month                         0.062361
# year                          0.058674
# shop_city_encoded             0.040425
# shop_type_encoded             0.034077
# item_price_mean_lag_1         0.033627
# first_open_month_num          0.023481
# item_cnt_month_lag_2          0.000000
# item_price_mean_lag_2         0.000000
# item_cnt_month_lag_3          0.000000
# item_price_mean_lag_3         0.000000
# item_cnt_month_lag_4          0.000000
# item_price_mean_lag_4         0.000000
# dtype: float32

from sklearn.model_selection import RandomizedSearchCV

grid_values_XGBR = {'n_estimators': list(np.arange(100, 1100, 100)), 'learning_rate': [0.001, 0.01, 0.1], 'max_depth': [3, 4, 5, 6, 7]}

fit_params={"early_stopping_rounds": 50, 
            "eval_metric" : "rmse", 
            "eval_set" : [[encoded_X_valid_scaled, y_valid]]}

XGBR_model = XGBRegressor(eval_metric="rmse", n_jobs = 8)
XGBR_model_mse = RandomizedSearchCV(XGBR_model, param_grid = grid_values_XGBR, scoring = 'neg_root_mean_squared_error', cv = None, verbose = 1, n_iter = 10)
XGBR_model_mse.fit(encoded_X_train_scaled, y_train, **fit_params)
print('Grid best parameter (min. neg_root_mean_squared_error): ', XGBR_model_mse.best_params_)



mean_squared_error(y_train, XGBR_model_mse.predict(encoded_X_train_scaled), squared = False)
mean_squared_error(y_valid, XGBR_model_mse.predict(encoded_X_valid_scaled), squared = False)


print('XGBRegressor with n_estimators = {}, learning_rate = {}, max_depth = {}: training set root-mean-squared error is {:.0f}'.format(XGBR_model_mse.best_params_['n_estimators'], XGBR_model_mse.best_params_['learning_rate'], XGBR_model_mse.best_params_['max_depth'], mean_squared_error(y_train, XGBR_model_mse.predict(encoded_X_train_scaled), squared = False)))
print('XGBRegressor with n_estimators = {}, learning_rate = {}, max_depth = {}: validation set root-mean-squared error is {:.0f}'.format(XGBR_model_mse.best_params_['n_estimators'], XGBR_model_mse.best_params_['learning_rate'], XGBR_model_mse.best_params_['max_depth'], mean_squared_error(y_valid, XGBR_model_mse.predict(encoded_X_valid_scaled), squared = False)))


pd.Series(XGBR_model.feature_importances_, index = encoded_X_train_scaled.columns).sort_values(ascending = False)



XGBR_best = XGBRegressor(n_estimators = 950, learning_rate = 0.01, max_depth = 5,  eval_metric="rmse", n_jobs = 8)
XGBR_best.fit(encoded_X_train_scaled, y_train)
mean_squared_error(y_valid, XGBR_best.predict(encoded_X_valid_scaled), squared = False)

from pickle import dump
# save the model
dump(XGBR_best, open('XGBR_best.pkl', 'wb'))

my_pred = pd.Series(XGBR_best.predict(encoded_X_test_scaled).clip(0,20), name = 'item_cnt_month', index = test_df_all['ID'])
# test score 1.69278
my_pred.to_csv('my_submission_XGBR_best.csv')



plt.figure()
plt.scatter(np.array(list(y_train.index)), np.array(list(y_train.values)))
plt.scatter(np.array(list(y_train.index)), XGBR_best.predict(encoded_X_train_scaled))

plt.figure()
plt.scatter(np.array(list(y_valid.index)), np.array(list(y_valid.values)))
plt.scatter(np.array(list(y_valid.index)), XGBR_best.predict(encoded_X_valid_scaled))






#################################

from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(random_state = 0, n_jobs = -1)
RFR.fit(encoded_X_train_scaled, y_train)
print(mean_squared_error(y_train, RFR.predict(encoded_X_train_scaled), squared = False))
print(mean_squared_error(y_valid, RFR.predict(encoded_X_valid_scaled), squared = False))
# 1.6746011397793024
# 1.933731006912131

pd.Series(RFR.feature_importances_, index = encoded_X_train_scaled.columns).sort_values(ascending = False)
# item_cnt_month_lag_4          0.137339
# item_cnt_month_lag_3          0.135974
# item_cnt_month_lag_2          0.132093
# item_cnt_month_lag_1          0.125347
# month                         0.085334
# shop_city_encoded             0.077291
# item_main_category_encoded    0.055901
# item_price_mean_lag_4         0.045825
# item_price_mean_lag_1         0.045752
# item_price_mean_lag_2         0.045507
# item_price_mean_lag_3         0.045427
# shop_type_encoded             0.038238
# year                          0.027514
# first_open_month_num          0.002458
# dtype: float64


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = np.arange(100, 1050, 50)
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [10, 20, 40, 60, 80, 100, None]
# Minimum number of samples required to split a node
min_samples_split = [2, 8, 16, 128, 1024, 4096]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 4, 128, 512, 1024, 4096]

# Create the random grid
random_grid_RFR = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


RFR_model = RandomForestRegressor(random_state = 0, n_jobs = -1)
RFR_model_rmse = RandomizedSearchCV(RFR_model, param_distributions = random_grid_RFR, scoring = 'neg_root_mean_squared_error', cv = None, verbose = 1, n_iter = 40)
RFR_model_rmse.fit(encoded_X_train_scaled, y_train)
print('Grid best parameter (min. neg_root_mean_squared_error): ', RFR_model_rmse.best_params_)
# Grid best parameter (min. neg_root_mean_squared_error):  {'n_estimators': 650, 'min_samples_split': 128, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 60}

RFR_best = RandomForestRegressor(n_estimators = 650, max_depth = 60, max_features= 'sqrt', min_samples_leaf = 4, min_samples_split = 128, random_state = 0, n_jobs = -1)
RFR_best.fit(encoded_X_train_scaled, y_train)

print(mean_squared_error(y_train, RFR_best.predict(encoded_X_train_scaled), squared = False))
print(mean_squared_error(y_valid, RFR_best.predict(encoded_X_valid_scaled), squared = False))
# 2.074137686378509
# 1.8754997033673564

pd.Series(RFR_best.feature_importances_, index = encoded_X_train_scaled.columns).sort_values(ascending = False)
# item_cnt_month_lag_2          0.195043
# item_cnt_month_lag_4          0.187127
# item_cnt_month_lag_3          0.186041
# item_cnt_month_lag_1          0.181715
# item_main_category_encoded    0.060592
# month                         0.042649
# item_price_mean_lag_3         0.027090
# item_price_mean_lag_4         0.025777
# item_price_mean_lag_2         0.025585
# item_price_mean_lag_1         0.024879
# shop_city_encoded             0.021820
# year                          0.012249
# shop_type_encoded             0.008223
# first_open_month_num          0.001211
# dtype: float64

plt.figure()
plt.scatter(np.array(list(y_train.index)), np.array(list(y_train.values)))
plt.scatter(np.array(list(y_train.index)), RFR_best.predict(encoded_X_train_scaled))

plt.figure()
plt.scatter(np.array(list(y_valid.index)), np.array(list(y_valid.values)))
plt.scatter(np.array(list(y_valid.index)), RFR_best.predict(encoded_X_valid_scaled))

from pickle import dump
# save the model
dump(RFR_best, open('RFR_best.pkl', 'wb'))


# from pickle import load
# RFR_best = load(open('RFR_best.pkl', 'rb'))

my_pred = pd.Series(RFR_best.predict(encoded_X_test_scaled).clip(0,20), name = 'item_cnt_month', index = test_df_all['ID'])
# test score 1.69278
my_pred.to_csv('my_submission_RFR_best.csv')

#%% save data

encoded_X_train.to_csv('encoded_X_train.csv')
encoded_X_train_scaled.to_csv('encoded_X_train_scaled.csv')
y_train.to_csv('y_train.csv')
encoded_X_valid.to_csv('encoded_X_valid.csv')
encoded_X_valid_scaled.to_csv('encoded_X_valid_scaled.csv')
y_valid.to_csv('y_valid.csv')
encoded_X_test.to_csv('encoded_X_test.csv')
encoded_X_test_scaled.to_csv('encoded_X_test_scaled.csv')










