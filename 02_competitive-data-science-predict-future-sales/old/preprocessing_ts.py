# -*- coding: utf-8 -*-
"""
Created on Sun Aug 01 21:50:00 2021

@author: Cedric Yu
"""

# import matplotlib.pyplot as plt
# import math
# import re


"""
# 
"""

#%% Preamble

import pandas as pd
# Make the output look better
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

import numpy as np
# import seaborn as sn
# import matplotlib.pyplot as plt

#%% info on shop_id and item_id; does not use training data yet
#%% shops: add columns for city and shop types

"""# in dataset_study.py, we did a fuzzy match on shop names. Looking also at their monthly sales count, we found that the following pairs are duplicates"""
fuzzy_match_pair = [(10, 11), (57, 0), (58, 1)]

"""# shop_id_city_type contains, for each of 60 shops, a map of shop_id to its city and shop type """
shop_id_city_type = pd.read_csv(r'shop_id_city_type.csv')
shop_id_city_type = shop_id_city_type[['shop_id', 'shop_city', 'shop_type']]


#%% items: add columns for item main category and platform

"""# items_id_cat_platform contains, for each of item, a map of item_id to its item_category_id, main category, and platform """
items_id_cat_platform = pd.read_csv('items_id_cat_platform.csv')
items_id_cat_platform['platform'].fillna('other', inplace = True)
items_id_cat_platform = items_id_cat_platform[['item_id', 'item_category_id', 'item_main_category', 'platform']]


#%% preprocessing

#%% lag (not advanced) features

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

#%% rolling features

""" # we create n-month rolling mean features from the <previous> n months, i.e. <not including> the current instance"""
""" # the following function assumes that df already contains the corresponding lag columns (with the same n) constructed by the lags function above"""

def rolling_mean(df, col, n=3): 
    df0 = df.copy()
    
    df0[col + '_rolling_' + str(n)] = df0[[col + '_lag_' + str(i+1) for i in range(n)]].mean(axis = 1)
    # df0[col + '_rolling_' + str(n)].fillna(0., inplace = True)
    
    return df0

#%% merge duplicate pairs from fuzzy_match_pair

""" # merge duplicate pairs """

# fuzzy_match_pair
# [(10, 11), (57, 0), (58, 1)]

def shop_duplicates(df) : 
    df0 = df.copy()
    for pair in fuzzy_match_pair : 
        df0['shop_id'].replace(pair[1], pair[0], inplace = True)
    return df0

#%% keep open shops and existing items only

""" # shops: <in the training set> (fit) (caution leakage) open for at least 3 months, open last month, fraction of which month they first opened"""
""" # item: <in the training set> (caution leakage) sold for at least 3 months, and sold in last 3 months, fraction of which month they were first sold if sold at all"""

""" # in the training set (fit), only keep the shops that are still open (defined above), items that are still sold"""

""" # validation/test sets: add 'first_open_month_num' for each shop; if it is a new shop, set it to one (i.e. assume it first opened at the end of training period)"""

from sklearn.base import BaseEstimator, TransformerMixin


class good_shops_items0(BaseEstimator, TransformerMixin) : 
    def __init__(self): # This will be called when the ColumnTransformer is called
        # print('__init__ is called.\n')
        pass
    
    def fit(self, X, y) : 
        
        df_ = pd.merge(X, y, how = 'left', left_index = True, right_index = True)
        
        def shop_open(df) :
            cnt_shop = df.pivot_table(values = 'item_cnt_month', index = 'shop_id', columns = 'date_block_num', aggfunc = [np.nansum])
            cnt_shop.fillna(0, inplace = True)
            cnt_shop.reset_index(inplace = True)
            cnt_shop.columns = cnt_shop.columns.get_level_values(1)
            cnt_shop.rename(columns = {'': 'shop_id'}, inplace = True)
            cnt_shop.set_index('shop_id', inplace = True)
            
            def open_last_month(row) : 
                if (row > 0).sum() < 3 or (row.iloc[-1:].sum() < 1.) : 
                    row['open_last_month'] = False
                else : row['open_last_month'] = True
                
                return row
                
            
            cnt_shop_still_open = cnt_shop.apply(open_last_month, axis = 1)
            cnt_shop_still_open = cnt_shop_still_open['open_last_month'].astype(int)
            
            def first_open_month(row) : 
                row['first_open_month_num'] = (row[row > 0].index[0] - row.index[0]) / (row.index[-1] - row.index[0])
                return row
            
            cnt_shop_first_open = cnt_shop.apply(first_open_month, axis = 1)
            cnt_shop_first_open = cnt_shop_first_open['first_open_month_num']
            
            cnt_shop_open = pd.merge(cnt_shop_still_open, cnt_shop_first_open, how = 'inner', left_index = True, right_index = True)
            
            return cnt_shop_open
        
        self.shop_open = shop_open(df_)
        
        def item_sold(df) : 
            
            cnt_item = df.pivot_table(values = 'item_cnt_month', index = 'item_id', columns = 'date_block_num', aggfunc = [np.nansum])
            cnt_item.fillna(0, inplace = True)
            cnt_item.reset_index(inplace = True)
            cnt_item.columns = cnt_item.columns.get_level_values(1)
            cnt_item.rename(columns = {'': 'item_id'}, inplace = True)
            cnt_item.set_index('item_id', inplace = True)    
            
            
            def sold_last_three_months(row) : 
                if (row > 0).sum() < 3 or (row.iloc[-3:].sum() < 1.) : 
                    row['sold_last_three_months'] = False
                else : row['sold_last_three_months'] = True
                
                return row
                
            
            cnt_item_still_sold = cnt_item.apply(sold_last_three_months, axis = 1)
            cnt_item_still_sold = cnt_item_still_sold['sold_last_three_months'].astype(int)
            
            def first_sold_month(row) : 
                if len(row[row > 0]) > 0 : 
                    row['first_sold_month_num'] = (row[row > 0].index[0] - row.index[0]) / (row.index[-1] - row.index[0])
                else : 
                    row['first_sold_month_num'] = -1
                return row
            
            cnt_item_first_sold = cnt_item.apply(first_sold_month, axis = 1)
            cnt_item_first_sold = cnt_item_first_sold['first_sold_month_num'].astype(int)
            
            cnt_item_sold = pd.merge(cnt_item_still_sold, cnt_item_first_sold, how = 'inner', left_index = True, right_index = True)
            
            return cnt_item_sold
        
        self.item_sold = item_sold(df_)
        
        df_ = pd.merge(df_, self.shop_open, how = 'left', on = 'shop_id')
        df_ = pd.merge(df_, self.item_sold, how = 'left', on = 'item_id')
        df_ = df_[(df_['open_last_month'] == 1) & (df_['sold_last_three_months'] == 1)]
        df_.drop(['open_last_month', 'sold_last_three_months', 'first_sold_month_num'], axis = 1, inplace = True)
        
        return df_.drop(['item_cnt_month'], axis = 1), df_['item_cnt_month']
    
    def transform(self, X) : 
        
        X_ = X.copy()
        
        X_ = pd.merge(X_, self.shop_open, how = 'left', on = 'shop_id')
        X_.drop(['open_last_month'], axis = 1, inplace = True)
        # first_open_month_num has NaN; those shops were not open yet. Assume they are open in the end of the training period
        X_['first_open_month_num'].fillna(1., inplace = True)
        
        return X_



#%% item mean price for future datasets

""" # (not used) make an educated guess that the monthly mean price of each item in the test set period (Nov 2015) is the mean price in the last month. """
""" # we will use lagged mean item price instead"""

class train_item_price_mean0(BaseEstimator, TransformerMixin) : 
    def __init__(self): # This will be called when the ColumnTransformer is called
        # print('__init__ is called.\n')
        pass
    
    def fit(self, X) : 
        
        X_ = X[['item_price_mean', 'date_block_num', 'shop_city', 'item_main_category']].copy()
        
        # take the mean prices of last month
        X_ = X_[X_['date_block_num'] == X_['date_block_num'].max()]
        # last month's mean price by city and item main category
        X_ = X_.groupby(['shop_city', 'item_main_category']).agg({'item_price_mean': np.nanmean})
        # X_.reset_index().drop(['index', 'date_block_num'], axis = 1, inplace = True)
        self.last_mean_price_city_cat = X_
        
    
    def transform(self, X) : 
        
        X_ = X.copy()
        
        X_ = pd.merge(X_, self.last_mean_price_city_cat, how = 'left', on = ['shop_city', 'item_main_category'])
        
        X_['item_price_mean'].fillna(X_['item_price_mean'].mean(), inplace = True)
        
        return X_

#%% mean (averaged over years) monthly total item count by shop and by item category

""" # computes, from the <training> data, the mean monthly item counts and prices, averaged over the years"""
""" # these two features appear (in hindsight) to give larger error; not used"""

class item_cnt_year_avg0(BaseEstimator, TransformerMixin) : 
    def __init__(self): # This will be called when the ColumnTransformer is called
        # print('__init__ is called.\n')
        pass
    
    def fit(self, X, y) : 
        
        df_ = pd.merge(X, y, how = 'left', left_index = True, right_index = True)
        
        df1 = df_.groupby(['shop_id', 'month', 'year']).agg({'item_cnt_month': np.nansum})
        df1 = df1.groupby(['shop_id', 'month']).agg({'item_cnt_month': np.nanmean}).reset_index()
        df1.rename(columns = {'item_cnt_month': 'item_cnt_month_by_shop'}, inplace = True)
        
        df2 = df_.groupby(['item_id', 'month', 'year']).agg({'item_cnt_month': np.nansum})
        df2 = df2.groupby(['item_id', 'month']).agg({'item_cnt_month': np.nanmean}).reset_index()
        df2.rename(columns = {'item_cnt_month': 'item_cnt_month_by_item'}, inplace = True)
        
        self.avg_cnt_month_shop = df1
        self.avg_cnt_month_item = df2
        
    
    def transform(self, X) : 
        
        X_ = X.copy()
        
        X_ = pd.merge(X_, self.avg_cnt_month_shop, how = 'left', on = ['shop_id', 'month'])
        X_ = pd.merge(X_, self.avg_cnt_month_item, how = 'left', on = ['item_id', 'month'])
        
        X_['item_cnt_month_by_shop'].fillna(X_['item_cnt_month_by_shop'].mean(), inplace = True)
        X_['item_cnt_month_by_item'].fillna(X_['item_cnt_month_by_item'].mean(), inplace = True)
        
        return X_



#%% encode categorical features
# from sklearn.base import BaseEstimator, TransformerMixin

""" # frequency encoder: encode categorical features by fractional value counts in the training dataset """
""" # we wrote (commented out) a mean (target) encoder in the end of this script """

class FrequencyEncoder(BaseEstimator, TransformerMixin) : 
    def __init__(self): # This will be called when the ColumnTransformer is called
        # print('__init__ is called.\n')
        self.encode_dict = {}
        self.col_to_encode = []
    
    def fit(self, X, col_to_encode = []) : 
        
        if len(col_to_encode) == 0 : 
            pass
        
        self.col_to_encode = col_to_encode.copy()
        X_cat = X[self.col_to_encode].copy()
        
        for col in self.col_to_encode : 
            col_freq = pd.DataFrame(X_cat[col].value_counts() / X_cat.size)
            col_freq = col_freq.reset_index()
            col_freq.rename(columns = {'index': col, col: col + '_encoded'}, inplace = True)
            
            self.encode_dict[col] = col_freq
        
    
    def transform(self, X) : 
        
        if len(self.col_to_encode) == 0 : 
            pass
        
        X_ = X.copy()
        # print('X is copied')
        for col in self.col_to_encode : 
            
            encoded_col = pd.merge(pd.DataFrame(X_[col]), 
                                   self.encode_dict[col], 
                                   how = 'left', on = col).set_index(X_.index)
            # print(encoded_col.head())
            # print(encoded_col.shape)
            encoded_col[col + '_encoded'].fillna(0, inplace = True)
            X_[col + '_encoded'] = encoded_col[col + '_encoded']
            # print(X_[col + '_encoded'].head())
            X_.drop(col, axis = 1, inplace = True)            
        
        return X_




#%% scaler

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()

# def scaler_fit_transform(df) : 
#     return scaler.fit_transform(df)

# def scaler_transform(df) : 
#     return scaler.transform(df)

#%% convert 'date' to datetime

# from datetime import datetime
# def date_datetime(row):
#     row['date']=datetime.strptime(row['date'], "%d.%m.%Y")
#     return row
# train_df_head = train_df_head.apply(date_datetime, axis='columns')
# # 'date' columm dtype: datetime64[ns]
# train_df_raw = train_df_raw.apply(date_datetime, axis='columns')



#%% target (mean) encoding of shop_id and item_id

# col_to_encode = ['shop_id', 'item_id']

# from sklearn.base import BaseEstimator, TransformerMixin

# class mean_encoder(BaseEstimator, TransformerMixin) : 
#     def __init__(self): # This will be called when the ColumnTransformer is called
#         # print('__init__ is called.\n')
#         self.encode_dict = {}
    
#     def fit(self, X, y, col_to_encode = []) : 
        
#         if len(col_to_encode) == 0 : 
#             pass
        
#         X_cat = X[col_to_encode].copy()
#         y_ = y.copy()
#         df_Xy_encode = pd.merge(X_cat[col_to_encode], y_, how='left', left_index=True, right_index=True)
        
        
#         for col in col_to_encode : 
#             df_Xy_encode2 = df_Xy_encode.groupby(col).agg({df_Xy_encode.columns[-1]: np.mean})
            
#             self.encode_dict[col] = df_Xy_encode2.to_dict()[df_Xy_encode.columns[-1]]
        
    
#     def transform(self, X, col_to_encode = []) : 
        
#         if len(col_to_encode) == 0 : 
#             pass
        
#         X_ = X.copy()
#         X_cat = X[col_to_encode].copy()
        
#         for col in col_to_encode : 

#             def col_encode_dict(x) : 
#                 if x in self.encode_dict[col] : 
#                     return self.encode_dict[col][x]
#                 return 0
            
#             df_X_with_encode = X_cat[col]
#             df_X_with_encode = df_X_with_encode.apply(col_encode_dict)
#             X_['encoded_' + col] = df_X_with_encode
#             X_.drop(col, axis = 1, inplace = True)
        
#         return X_

# my_mean_encoder = mean_encoder()

# my_mean_encoder.fit(X_train, y_train, col_to_encode)
# encoded_X_train = my_mean_encoder.transform(X_train, col_to_encode)
# encoded_X_valid = my_mean_encoder.transform(X_valid, col_to_encode)

# encoded_X_train.shape  # (2610004, 4)
# encoded_X_valid.shape  # (325845, 4)

# #%% groupby monthly per-item, per-shop data

# df_train2 = pd.merge(encoded_X_train, y_train, how='left', left_index= True, right_index=True)
# df_train2 = df_train2.groupby(['date_block_num',  'encoded_shop_id',  'encoded_item_id']).agg({'mean_item_price_month': np.mean, 'item_cnt_month': np.mean})
# # df_train2.shape  # (1284851, 2)


# df_valid2 = pd.merge(encoded_X_valid, y_valid, how='left', left_index= True, right_index=True)
# df_valid2 = df_valid2.groupby(['date_block_num',  'encoded_shop_id',  'encoded_item_id']).agg({'mean_item_price_month': np.mean, 'item_cnt_month': np.mean})
# # df_valid2.shape   # (145562, 2)

# encoded_X_train = df_train2.drop(['item_cnt_month'], axis = 1)
# y_train = df_train2['item_cnt_month']

# encoded_X_valid = df_valid2.drop(['item_cnt_month'], axis = 1)
# y_valid = df_valid2['item_cnt_month']

