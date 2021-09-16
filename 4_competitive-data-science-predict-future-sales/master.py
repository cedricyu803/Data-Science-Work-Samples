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
# pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None  # default='warn' # ignores warning about dropping columns inplace
import numpy as np
import matplotlib.pyplot as plt
# import re

import re
import os
os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\4_competitive-data-science-predict-future-sales')

#%% load engineered datasets

X_train_encoded4_scaled = pd.read_csv('engineered_datasets/X_train_encoded4_scaled.csv', index_col = [0])
X_valid_encoded4_scaled = pd.read_csv('engineered_datasets/X_valid_encoded4_scaled.csv', index_col = [0])
X_test_encoded4_scaled = pd.read_csv('engineered_datasets/X_test_encoded4_scaled.csv', index_col = [0])

y_train = pd.read_csv('engineered_datasets/y_train.csv', index_col = [0]).squeeze()
y_valid = pd.read_csv('engineered_datasets/y_valid.csv', index_col = [0]).squeeze()

y_test = pd.read_csv('engineered_datasets/y_test.csv', index_col = [0])
y_test['ID'] = y_test['ID'].astype(int)

import gc 
gc.collect()

#%% clip item_cnt_month in the range [0, 20]

def clip_cnt(y) : 
    y = y.copy()
    y.clip(0., 20., inplace = True)
    return y


#%% model scores

from sklearn.metrics import mean_squared_error

#################################
"""# LinearRegression """

# from sklearn.linear_model import LinearRegression

# linreg = LinearRegression(n_jobs = -1)
# linreg.fit(encoded_X_train_scaled, y_train)

# print('Linear Regression: training set root-mean-squared error is {:.0f}'.format(mean_squared_error(y_train, linreg.predict(encoded_X_train_scaled), squared = False)))
# print('Linear Regression: validation set root-mean-squared error is {:.0f}'.format(mean_squared_error(y_valid, linreg.predict(encoded_X_valid_scaled), squared = False)))
# # Linear Regression: training set root-mean-squared error is 3
# # Linear Regression: validation set root-mean-squared error is 2


# plt.figure()
# plt.scatter(np.array(list(y_train.index)), np.array(list(y_train.values)))
# plt.scatter(np.array(list(y_train.index)), linreg.predict(encoded_X_train))

# plt.figure()
# plt.scatter(np.array(list(y_valid.index)), np.array(list(y_valid.values)))
# plt.scatter(np.array(list(y_valid.index)), linreg.predict(encoded_X_valid))

#################################
"""# Ridge """

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

"""# Lasso """

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

"""# Polynomial """

# from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(degree=3)
# encoded_X_train_scaled_poly = poly.fit_transform(encoded_X_train_scaled)
# encoded_X_valid_scaled_poly = poly.transform(encoded_X_valid_scaled)

# from sklearn.linear_model import Ridge
# linridge_poly = Ridge(alpha = 100.)
# linridge_poly.fit(encoded_X_train_scaled_poly, y_train)
# print('Ridge Polynomial (p = 3) Regression with alpha = {}: training set root-mean-squared log error is {:.0f}'.format(100., mean_squared_error(y_train, linridge_poly.predict(encoded_X_train_scaled_poly), squared = False)))
# print('Ridge Polynomial (p = 3) Regression with alpha = {}: validation set root-mean-squared log error is {:.0f}'.format(100., mean_squared_error(y_valid, linridge_poly.predict(encoded_X_valid_scaled_poly), squared = False)))
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

"""# knn """

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

# !!!
"""# XGBRegressor """

from xgboost import XGBRegressor

XGBR_model = XGBRegressor(eval_metric = "rmse", 
                          learning_rate = 0.05, 
                          max_depth = 8,
                          n_estimators = 650,
                          reg_lambda = 0.9,
                          n_jobs = 6)
XGBR_model.fit(X_train_encoded4_scaled, y_train, eval_set = [(X_train_encoded4_scaled, y_train), (X_valid_encoded4_scaled, y_valid)], early_stopping_rounds = 50)
# [157]	validation_0-rmse:1.88953	validation_1-rmse:1.99079

# print(mean_squared_error(y_train, XGBR_model.predict(X_train_encoded4_scaled), squared = False))
# print(mean_squared_error(y_valid, XGBR_model.predict(X_valid_encoded4_scaled), squared = False))


XGBR_feature_importances = pd.Series(XGBR_model.feature_importances_, index = original_cols).sort_values(ascending = False)
# item_cnt_month_lag_1                 0.589584
# item_main_category_freq_encoded      0.102044
# platform_freq_encoded                0.100358
# year_freq_encoded                    0.070516
# month_freq_encoded                   0.046233
# shop_city_freq_encoded               0.034644
# item_price_mean_lag_1                0.033968
# shop_type_freq_encoded               0.022653
# item_cnt_month_lag_4                 0.000000
# item_price_mean_lag_14               0.000000
# item_cnt_month_lag_3                 0.000000
# first_open_month_num_freq_encoded    0.000000
# item_price_mean_lag_13               0.000000
# item_cnt_month_lag_12                0.000000
# item_price_mean_lag_12               0.000000
# item_cnt_month_lag_2                 0.000000
# item_price_mean_lag_3                0.000000
# item_price_mean_lag_2                0.000000
# item_cnt_month_lag_14                0.000000
# item_cnt_month_lag_13                0.000000
# item_price_mean_lag_4                0.000000
# dtype: float32

import seaborn as sns
fig = plt.figure('feature_importances', figsize = (10, 6), dpi = 200)
sns.barplot(y = XGBR_feature_importances.iloc[:18].index, x = XGBR_feature_importances.iloc[:18].values, color = 'skyblue')
# LGBMreg_feature_importances.plot.barh()
ax = plt.gca()
ax.set_xlabel('Feature importance')
ax.set_ylabel(None)
ax.set_yticklabels(ax.get_yticklabels(), fontsize = 'small')
ax.set_title('XGBM Regressor')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.xlim(0,1)
fig.tight_layout()
# plt.savefig('plots/XGBM_feature_importances.png', dpi = 200)

XGBRpred = XGBR_model.predict(X_test_encoded4_scaled)
y_pred = y_test.copy()

y_pred['item_cnt_month'] = XGBRpred
y_pred = y_pred.drop(['shop_id', 'item_id'], axis = 1).set_index(['ID'])
y_pred = clip_cnt(y_pred)
y_pred.to_csv('y_pred_XGBM.csv')


""" RandomizedSearchCV"""
from sklearn.model_selection import RandomizedSearchCV

grid_values_XGBR = {'n_estimators': list(np.arange(100, 1100, 100)), 'learning_rate': [0.001, 0.01, 0.1], 'max_depth': [3, 4, 5, 6, 7]}

fit_params={"early_stopping_rounds": 50, 
            "eval_metric" : "rmse", 
            "eval_set" : [[X_valid_encoded4_scaled, y_valid]]}

XGBR_model = XGBRegressor(eval_metric="rmse", n_jobs = 6)
XGBR_model_mse = RandomizedSearchCV(XGBR_model, param_distributions = grid_values_XGBR, scoring = 'neg_mean_squared_error', cv = None, verbose = 1, n_iter = 10)
XGBR_model_mse.fit(X_train_encoded4_scaled, y_train, **fit_params)
print('Grid best parameter (min. neg_root_mean_squared_error): ', XGBR_model_mse.best_params_)



mean_squared_error(y_train, XGBR_model_mse.predict(X_train_encoded4_scaled), squared = False)
mean_squared_error(y_valid, XGBR_model_mse.predict(X_valid_encoded4_scaled), squared = False)


# from pickle import dump
# # save the model
# dump(XGBR_best, open('XGBR_best.pkl', 'wb'))

# my_pred = pd.Series(XGBR_best.predict(encoded_X_test_scaled).clip(0,20), name = 'item_cnt_month', index = test_df_all['ID'])

# my_pred.to_csv('my_submission_XGBR_best.csv')


#################################

"""# RandomForestRegressor """

from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators = 1000, 
                            max_depth = 60, 
                            max_features= 'sqrt', 
                            min_samples_leaf = 4, 
                            min_samples_split = 128, 
                            random_state = 0, n_jobs = -1)
RFR.fit(X_train_encoded4_scaled, y_train)

print(mean_squared_error(y_train, RFR.predict(X_train_encoded4_scaled), squared = False))
print(mean_squared_error(y_valid, RFR.predict(X_valid_encoded4_scaled), squared = False))
# 1.9423862434612256
# 1.9735567543499082

RFRfeature_imp = pd.Series(RFR.feature_importances_, index = X_train_encoded4_scaled.columns).sort_values(ascending = False)

# item_cnt_month_lag_12                0.113806
# item_cnt_month_lag_2                 0.113337
# item_cnt_month_lag_14                0.107384
# item_cnt_month_lag_13                0.105285
# item_cnt_month_lag_3                 0.103727
# item_cnt_month_lag_4                 0.101463
# item_cnt_month_lag_1                 0.099316
# platform_freq_encoded                0.048903
# item_main_category_freq_encoded      0.044323
# month_freq_encoded                   0.028175
# shop_city_freq_encoded               0.022603
# item_price_mean_lag_4                0.014276
# item_price_mean_lag_13               0.013818
# item_price_mean_lag_12               0.013179
# item_price_mean_lag_3                0.012860
# item_price_mean_lag_2                0.012674
# item_price_mean_lag_1                0.012668
# item_price_mean_lag_14               0.012471
# year_freq_encoded                    0.011408
# shop_type_freq_encoded               0.008325
# first_open_month_num_freq_encoded    0.000000
# dtype: float64



import seaborn as sns
fig = plt.figure('feature_importances', figsize = (10, 6), dpi = 200)
sns.barplot(y = RFRfeature_imp.iloc[:18].index, x = RFRfeature_imp.iloc[:18].values, color = 'skyblue')
# LGBMreg_feature_importances.plot.barh()
ax = plt.gca()
ax.set_xlabel('Feature importance')
ax.set_ylabel(None)
ax.set_yticklabels(ax.get_yticklabels(), fontsize = 'small')
ax.set_title('Random Forest Regressor')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.xlim(0,1)
fig.tight_layout()
# plt.savefig('plots/RFR_feature_importances.png', dpi = 200)



RFRpred = RFR.predict(X_test_encoded4_scaled)
y_pred = y_test.copy()
y_pred['item_cnt_month'] = RFRpred
y_pred = y_pred.drop(['shop_id', 'item_id'], axis = 1).set_index(['ID'])
y_pred = clip_cnt(y_pred)
y_pred.to_csv('y_pred_RFR.csv')
# 1.82240


# for best score ever, 1.50672
RFRfeature_imp_best_old = pd.read_csv('old/RFR_old_best_feat.csv', usecols=[0,2], header = None, index_col = [0]).squeeze()

import seaborn as sns
fig = plt.figure('feature_importances', figsize = (10, 6), dpi = 200)
sns.barplot(y = RFRfeature_imp_best_old.index, x = RFRfeature_imp_best_old.values, color = 'skyblue')
# LGBMreg_feature_importances.plot.barh()
ax = plt.gca()
ax.set_xlabel('Feature importance')
ax.set_ylabel(None)
ax.set_yticklabels(ax.get_yticklabels(), fontsize = 'small')
ax.set_title('Random Forest Regressor')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.xlim(0,1)
fig.tight_layout()
# plt.savefig('plots/RFR_feature_importances_old_best.png', dpi = 200)




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
RFR_model_rmse = RandomizedSearchCV(RFR_model, param_distributions = random_grid_RFR, scoring = 'neg_mean_squared_error', cv = None, verbose = 1, n_iter = 6)
RFR_model_rmse.fit(X_train_encoded4_scaled, y_train)
print('Grid best parameter (min. neg_root_mean_squared_error): ', RFR_model_rmse.best_params_)
# Grid best parameter (min. neg_root_mean_squared_error):  {'n_estimators': 650, 'min_samples_split': 128, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 60}

RFR_best = RandomForestRegressor(n_estimators = 650, max_depth = 60, max_features= 'sqrt', min_samples_leaf = 4, min_samples_split = 128, random_state = 0, n_jobs = -1)
RFR_best.fit(X_train_encoded4_scaled, y_train)

print(mean_squared_error(y_train, RFR_best.predict(X_train_encoded4_scaled), squared = False))
print(mean_squared_error(y_valid, RFR_best.predict(X_valid_encoded4_scaled), squared = False))
# 2.074137686378509
# 1.8754997033673564

pd.Series(RFR_best.feature_importances_, index = X_train_encoded4_scaled.columns).sort_values(ascending = False)


# from pickle import dump
# # save the model
# dump(RFR_best, open('RFR_best.pkl', 'wb'))


# from pickle import load
# RFR_best = load(open('RFR_best.pkl', 'rb'))

# my_pred = pd.Series(RFR_best.predict(X_test_encoded4_scaled).clip(0,20), name = 'item_cnt_month', index = test_df_all['ID'])

# my_pred.to_csv('my_submission_RFR_best.csv')



#################################

"""# lightGBM"""

import lightgbm as lgb

# lgbm does not support funny column names
original_cols = X_train_encoded4_scaled.columns
X_train_encoded4_scaled.columns = np.arange(0,len(original_cols))
X_valid_encoded4_scaled.columns = np.arange(0,len(original_cols))
X_test_encoded4_scaled.columns = np.arange(0,len(original_cols))


LGBMreg = lgb.LGBMRegressor(boosting_type = 'gbdt', 
                            learning_rate = 0.02, 
                            num_leaves = 800,
                            n_estimators = 1000, 
                            num_iterations = 5000, 
                            max_bin = 500, 
                            feature_fraction = 0.7, 
                            bagging_fraction = 0.7,
                            lambda_l2 = 0.5,
                            max_depth = 25,
                            silent = False
                            )

LGBMreg.fit(X_train_encoded4_scaled, y_train,
            eval_set=[(X_train_encoded4_scaled, y_train), (X_valid_encoded4_scaled, y_valid)], 
            eval_metric = 'rmse',
            early_stopping_rounds = 50, 
            verbose = True)
# training's rmse: 1.83872	training's l2: 3.38088	valid_1's rmse: 1.96716	valid_1's l2: 3.86971



# print(mean_squared_error(y_train, LGBMreg.predict(X_train_encoded4_scaled), squared = False))
# print(mean_squared_error(y_valid, LGBMreg.predict(X_valid_encoded4_scaled), squared = False))


LGBMreg_feature_importances = pd.Series(LGBMreg.feature_importances_, index = original_cols).sort_values(ascending = False)
LGBMreg_feature_importances = LGBMreg_feature_importances / LGBMreg_feature_importances.max()

# item_price_mean_lag_1                1.000000
# month_freq_encoded                   0.553917
# shop_city_freq_encoded               0.394655
# item_price_mean_lag_2                0.382292
# item_cnt_month_lag_1                 0.264732
# shop_type_freq_encoded               0.262482
# platform_freq_encoded                0.198055
# item_main_category_freq_encoded      0.177397
# item_cnt_month_lag_2                 0.099561
# item_price_mean_lag_3                0.069268
# year_freq_encoded                    0.062109
# item_price_mean_lag_4                0.033498
# item_cnt_month_lag_3                 0.023089
# item_cnt_month_lag_4                 0.001568
# item_cnt_month_lag_14                0.000000
# item_cnt_month_lag_13                0.000000
# item_price_mean_lag_12               0.000000
# item_price_mean_lag_13               0.000000
# item_price_mean_lag_14               0.000000
# first_open_month_num_freq_encoded    0.000000
# item_cnt_month_lag_12                0.000000
# dtype: float64






import seaborn as sns

fig = plt.figure('feature_importances', figsize = (10, 6), dpi = 200)
sns.barplot(y = LGBMreg_feature_importances.iloc[:18].index, x = LGBMreg_feature_importances.iloc[:18].values, color = 'skyblue')
# LGBMreg_feature_importances.plot.barh()
ax = plt.gca()
ax.set_xlabel('Feature importance')
ax.set_ylabel(None)
ax.set_yticklabels(ax.get_yticklabels(), fontsize = 'small')
ax.set_title('lightGBM Regressor')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(0,1)
fig.tight_layout()
# plt.savefig('plots/lightgbm_feature_importances.png', dpi = 200)

# from pickle import dump
# # save the model
# dump(LGBMreg, open('LGBMreg.pkl', 'wb'))

LGBMpred = LGBMreg.predict(X_test_encoded4_scaled)
y_pred = y_test.copy()
y_pred['item_cnt_month'] = LGBMpred
y_pred = y_pred.drop(['shop_id', 'item_id'], axis = 1).set_index(['ID'])
y_pred = clip_cnt(y_pred)
y_pred.to_csv('y_pred_LGBM.csv')
# 1.73834


"""RandomizedSearchCV """
from sklearn.model_selection import RandomizedSearchCV

grid_values_LGBM = {'n_estimators': list(np.arange(100, 1100, 100)), 'learning_rate': [0.01], 'max_depth': list(np.arange(1, 26, 2)), 'max_bin': list(np.arange(100, 600, 100)), 'num_leaves': list(np.arange(100, 800, 100))}

fit_params_LGBM={"eval_metric" : "rmse", 
            "eval_set" : [[X_valid_encoded4_scaled, y_valid]]}

LGBMreg2 = lgb.LGBMClassifier(eval_metric="auc", n_jobs = 8)

LGBMreg_search = RandomizedSearchCV(LGBMreg2, param_distributions = grid_values_LGBM, scoring = 'neg_mean_squared_error', cv = None, verbose = 1, n_iter = 10)
LGBMreg_search.fit(X_train_encoded4_scaled, y_train, **fit_params_LGBM)
print('Grid best parameters (roc_auc): ', LGBMreg_search.best_params_)














