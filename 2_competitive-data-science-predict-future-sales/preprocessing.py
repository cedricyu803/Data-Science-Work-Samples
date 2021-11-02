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
# drop outliers: keep item_cnt_day in (0, 750), and item_price in (0.01, 75000)
# merge duplicate shop pairs
# aggregate monthly sales count and mean item prices in training set
# add 'month' and 'year' columns to both train + validation and test sets
# concatenate train + validation and test sets into a single df
# append shop_city, shop_type, 'item_category_id', 'item_main_category', 'platform' columns
# create 1-4, 12-14-month lag features for lag_cols = ['item_cnt_month', 'item_price_mean']

# we choose a time-ordered split of train and validation sets: train set spans March 2014 - July 2015, and validation set spans Aug - Oct 2015
# feature-target split

feature engineering: 

# train:

# only keep the shops that are still open (open for at least 3 months, open last month) and add column of the fraction of which month they first opened)
# only keep items that are still sold (sold for at least 3 months, and sold in last 3 months), add a column of the fraction of which month they were first sold if sold at all
# encode categorical features
# select features
# min-max scaling

# valid and test:

# merge duplicate shop pairs
# X_valid = pp.shop_duplicates(X_valid)
# add 'first_open_month_num' for each shop; if it is a new shop, set it to one (i.e. assume it first opened at the end of training period)
# encode categorical features
# select features
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

#%% import preprocessing functions
import preprocessing_util as pp

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


#%% drop outliers

train_df_month = train_df_raw.copy()

train_df_month = train_df_month[ 
    (train_df_month['item_cnt_day'] > 0) 
    & (train_df_month['item_cnt_day'] < 750) 
    & (train_df_month['item_price'] > 0.01) 
    & (train_df_month['item_price'] < 75000
       )]

# train_df_raw.shape # (2935849, 5)

#%% merge shop duplicate pairs

train_df_month = pp.shop_duplicates(train_df_month)

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
train_df_month_test = pd.merge(train_df_month_test, pp.shop_id_city_type, how = 'left', on = 'shop_id')
train_df_month_test = pd.merge(train_df_month_test, pp.items_id_cat_platform, how = 'left', on = 'item_id')

# train_df_month.shape
# (1608224, 7)

#%% create lag features from target

lag_months = 14
lag_cols = ['item_cnt_month', 'item_price_mean']

for col in lag_cols : 
    train_df_month_test = pp.lags(train_df_month_test, col, lag_months)

""" keep lag features: 1-3, 12-14 months for item_cnt_month, and 1-2, 11-12 months for item_price_mean"""
train_df_month_test = train_df_month_test.drop(['item_cnt_month_lag_' + str(i) for i in np.arange(5, 12)], axis = 1)
train_df_month_test = train_df_month_test.drop(['item_price_mean_lag_' + str(i) for i in np.arange(5, 12)], axis = 1)


#%% train-validation split

""" We used largest lag month = 14.  """
"""# we do a time-ordered split; training set contains March 2014 to Jul 2015, validation set contains Aug 2015 - Oct 2015"""

df_train = train_df_month_test[(train_df_month_test['date_block_num'] > lag_months - 1) & (train_df_month_test['date_block_num'] < 31 )]

df_valid = train_df_month_test[(train_df_month_test['date_block_num'] > 30) & (train_df_month_test['date_block_num'] < 34 )]

df_test = train_df_month_test[ train_df_month_test['date_block_num'] == 34 ]

#%% feature-target split

X_train = df_train.drop(['item_cnt_month', 'ID'], axis = 1)
y_train = df_train['item_cnt_month']
# print(list(X_train.columns))
# ['date_block_num', 'shop_id', 'item_id', 'item_price_mean', 'month', 'year', 'shop_city', 'shop_type', 'item_category_id', 'item_main_category', 'platform', 'item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3', 'item_cnt_month_lag_12', 'item_cnt_month_lag_13', 'item_cnt_month_lag_14', 'item_price_mean_lag_1', 'item_price_mean_lag_2', 'item_price_mean_lag_11', 'item_price_mean_lag_12']

# X_train.shape
# Out[32]: (725301, 21)

X_valid = df_valid.drop(['item_cnt_month', 'item_price_mean', 'ID'], axis = 1)
y_valid = df_valid['item_cnt_month']

# X_valid.shape
# Out[33]: (94645, 21)

X_test = df_test.drop(['item_cnt_month', 'item_price_mean', 'ID'], axis = 1)
y_test = df_test[['ID', 'shop_id', 'item_id', 'item_cnt_month']]

# X_test.shape
# Out[36]: (214200, 21)

#%% Add new shop and item columns

good_shops_items = pp.good_shops_items0()

X_train, y_train = good_shops_items.fit(X_train, y_train)
X_valid = good_shops_items.transform(X_valid)
X_test = good_shops_items.transform(X_test)

X_train.shape
# Out[34]: (547882, 22)
X_valid.shape
# Out[35]: (94645, 21)
X_test.shape
# Out[36]: (214200, 21)

""" check indices in X and y """

#%% encoding categorical features

onehot_cols1 = []

freq_cols2 = ['first_open_month_num', 'month', 'year', 'shop_city', 'shop_type', 'item_main_category', 'platform']

target_mean_cols3 = []


"""# one-hot encoding"""
# OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)# sparse=False # will return sparse matrix if set True else will return an array
# X_train_encoded1 = one_hot_fit_transform(X_train)
# X_valid_encoded1 = one_hot_transform(X_valid)
# X_test_encoded1 = one_hot_transform(X_test)

X_train_encoded1 = X_train
X_valid_encoded1 = X_valid
X_test_encoded1 = X_test


"""# frequency encoding"""
import category_encoders as ce

# change data type to object before feeding it into the encoder
X_train_encoded1[freq_cols2] = X_train_encoded1[freq_cols2].astype(object)
freq_encoder2 = ce.count.CountEncoder()
freq_encoder2.fit(X_train_encoded1[freq_cols2])

X_train_encoded2 = pd.concat([X_train_encoded1, freq_encoder2.transform(X_train_encoded1[freq_cols2]).rename(columns = dict(zip(freq_cols2, [col + '_freq_encoded' for col in freq_cols2])))], axis = 1).drop(freq_cols2, axis = 1)

X_valid_encoded1[freq_cols2] = X_valid_encoded1[freq_cols2].astype(object)
X_valid_encoded2 = pd.concat([X_valid_encoded1, freq_encoder2.transform(X_valid_encoded1[freq_cols2]).rename(columns = dict(zip(freq_cols2, [col + '_freq_encoded' for col in freq_cols2])))], axis = 1).drop(freq_cols2, axis = 1)

X_test_encoded1[freq_cols2] = X_test_encoded1[freq_cols2].astype(object)
X_test_encoded2 = pd.concat([X_test_encoded1, freq_encoder2.transform(X_test_encoded1[freq_cols2]).rename(columns = dict(zip(freq_cols2, [col + '_freq_encoded' for col in freq_cols2])))], axis = 1).drop(freq_cols2, axis = 1)
X_test_encoded2[[col + '_freq_encoded' for col in freq_cols2]] = X_test_encoded2[[col + '_freq_encoded' for col in freq_cols2]]


"""target encoding """
# change data type to object before feeding it into the encoder
# X_train_encoded2[target_mean_cols3] = X_train_encoded2[target_mean_cols3].astype(object)

# target_mean_encoder3 = ce.target_encoder.TargetEncoder()
# target_mean_encoder3.fit(X_train_encoded2[target_mean_cols3], y_train)

# X_train_encoded3 = pd.concat([X_train_encoded2, target_mean_encoder3.transform(X_train_encoded2[target_mean_cols3]).rename(columns = dict(zip(target_mean_cols3, [col + '_mean_encoded' for col in target_mean_cols3])))], axis = 1).drop(target_mean_cols3, axis = 1)


# X_valid_encoded2[target_mean_cols3] = X_valid_encoded2[target_mean_cols3].astype(object)
# X_valid_encoded3 = pd.concat([X_valid_encoded2, target_mean_encoder3.transform(X_valid_encoded2[target_mean_cols3]).rename(columns = dict(zip(target_mean_cols3, [col + '_mean_encoded' for col in target_mean_cols3])))], axis = 1).drop(target_mean_cols3, axis = 1)



# X_test_encoded2[target_mean_cols3] = X_test_encoded2[target_mean_cols3].astype(object)
# X_test_encoded3 = pd.concat([X_test_encoded2, target_mean_encoder3.transform(X_test_encoded2[target_mean_cols3]).rename(columns = dict(zip(target_mean_cols3, [col + '_mean_encoded' for col in target_mean_cols3])))], axis = 1).drop(target_mean_cols3, axis = 1)

X_train_encoded3 = X_train_encoded2
X_valid_encoded3 = X_valid_encoded2
X_test_encoded3 = X_test_encoded2

#%% fillna that resulted from encoding

X_valid_encoded3 = X_valid_encoded3.fillna(X_train_encoded3.mean())
X_test_encoded3 = X_test_encoded3.fillna(X_train_encoded3.mean())



#%% select columns

cols_to_keep = ['item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3',
       'item_cnt_month_lag_4', 'item_cnt_month_lag_12',
       'item_cnt_month_lag_13', 'item_cnt_month_lag_14',
       'item_price_mean_lag_1', 'item_price_mean_lag_2',
       'item_price_mean_lag_3', 'item_price_mean_lag_4',
       'item_price_mean_lag_12', 'item_price_mean_lag_13',
       'item_price_mean_lag_14', 'first_open_month_num_freq_encoded',
       'month_freq_encoded', 'year_freq_encoded', 'shop_city_freq_encoded',
       'shop_type_freq_encoded', 'item_main_category_freq_encoded',
       'platform_freq_encoded']

X_train_encoded4 = X_train_encoded3[cols_to_keep]
X_valid_encoded4 = X_valid_encoded3[cols_to_keep]
X_test_encoded4 = X_test_encoded3[cols_to_keep]


#%% scaler

from sklearn.preprocessing import MinMaxScaler
scaler5 = MinMaxScaler()

X_train_encoded4_scaled = scaler5.fit_transform(X_train_encoded4)
X_train_encoded4_scaled = pd.DataFrame(X_train_encoded4_scaled, columns = X_train_encoded4.columns, index = X_train_encoded4.index)

X_valid_encoded4_scaled = scaler5.transform(X_valid_encoded4)
X_valid_encoded4_scaled = pd.DataFrame(X_valid_encoded4_scaled, columns = X_valid_encoded4.columns, index = X_valid_encoded4.index)

X_test_encoded4_scaled = scaler5.transform(X_test_encoded4)
X_test_encoded4_scaled = pd.DataFrame(X_test_encoded4_scaled, columns = X_test_encoded4.columns, index = X_test_encoded4.index)

#%% clip item_cnt_month in the range [0, 20]

def clip_cnt(y) : 
    y = y.copy()
    y.clip(0., 20., inplace = True)
    return y

y_train = clip_cnt(y_train)
y_valid = clip_cnt(y_valid)
# train_df_month['item_cnt_month'].plot(kind = 'box')

#%% save pre-processed datasets

X_train_encoded4.to_csv('engineered_datasets/X_train_encoded4.csv')
X_valid_encoded4.to_csv('engineered_datasets/X_valid_encoded4.csv')
X_train_encoded4_scaled.to_csv('engineered_datasets/X_train_encoded4_scaled.csv')
X_valid_encoded4_scaled.to_csv('engineered_datasets/X_valid_encoded4_scaled.csv')
y_train.to_csv('engineered_datasets/y_train.csv')
y_valid.to_csv('engineered_datasets/y_valid.csv')

X_test_encoded4_scaled.to_csv('engineered_datasets/X_test_encoded4_scaled.csv')
y_test.to_csv('engineered_datasets/y_test.csv')
