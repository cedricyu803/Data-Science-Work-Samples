# -*- coding: utf-8 -*-
"""
Created on Sun Aug 01 11:00:00 2021

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
pd.options.mode.chained_assignment = None  # default='warn' # ignores warning about dropping columns inplace
import numpy as np
import matplotlib.pyplot as plt


#%% load dataset train.csv

# import original training dataset
train_df_raw = pd.read_csv(r'datasets\train.csv', low_memory=False)
train_df_raw.reset_index(inplace=True)
train_df_raw.drop('index', inplace=True, axis=1)
train_df_raw.shape
# train_df_raw.shape
# (1460, 81)

# separate features and labels
X_train_valid = train_df_raw.drop(['SalePrice'], axis = 1)
y_train_valid = train_df_raw['SalePrice']
X_train_valid.reset_index(inplace=True)
X_train_valid.drop('index', inplace=True, axis=1)

#%% investigate our data

# nan_col is a list of column names in X_train_valid with nan entries
nan_col = [ col_name for col_name in X_train_valid.columns if X_train_valid[col_name].isnull().sum() > 0 ]
# len(nan_col) # 19

# dictionary of the number of nan (if exist) in each column
num_nan = {}
for i in range(len(X_train_valid.columns)) : 
    if X_train_valid.iloc[:, i].isnull().sum() > 0 :
        num_nan[X_train_valid.columns[i]] = X_train_valid.iloc[:, i].isnull().sum()
# num_nan
# {'LotFrontage': 259,
#  'Alley': 1369,
#  'MasVnrType': 8,
#  'MasVnrArea': 8,
#  'BsmtQual': 37,
#  'BsmtCond': 37,
#  'BsmtExposure': 38,
#  'BsmtFinType1': 37,
#  'BsmtFinType2': 38,
#  'Electrical': 1,
#  'FireplaceQu': 690,
#  'GarageType': 81,
#  'GarageYrBlt': 81,
#  'GarageFinish': 81,
#  'GarageQual': 81,
#  'GarageCond': 81,
#  'PoolQC': 1453,
#  'Fence': 1179,
#  'MiscFeature': 1406}

# list of column names of numerical variables
numerical_col = [ col_name for col_name in X_train_valid.columns if X_train_valid[col_name].dtype in ['int64', 'float64'] ]
# numerical columns with nan: 
# [s for s in numerical_col if s in nan_col]
# ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']

# list of column names of categorical variables
categorical_col = [ col_name for col_name in X_train_valid.columns if X_train_valid[col_name].dtype in ['O'] ]

# accounted for all columns
# len(numerical_col) + len(categorical_col) # 80

# [s for s in categorical_col if s in nan_col]
# ['Alley',
#  'MasVnrType',
#  'BsmtQual',
#  'BsmtCond',
#  'BsmtExposure',
#  'BsmtFinType1',
#  'BsmtFinType2',
#  'Electrical',
#  'FireplaceQu',
#  'GarageType',
#  'GarageFinish',
#  'GarageQual',
#  'GarageCond',
#  'PoolQC',
#  'Fence',
#  'MiscFeature']

# dictionary of cardinality of each categorical column
categorical_cols_cardinality = {}
for col_name in categorical_col : 
    categorical_cols_cardinality[col_name] =  X_train_valid[col_name].nunique()
# Neighborhood, Exterior1st and Exterior2nd have large (> 10) cardinalities, but they look important so let's keep them

ordinal_encode_cols = ['Street', 'Alley', 'LotShape', 'Utilities', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']

OH_encode_cols = [ col_name for col_name in categorical_col if col_name not in ordinal_encode_cols ]

# accounted for all categorical columns
# len(label_encode_cols) + len(OH_encode_cols) == len(categorical_col) # True



"""
Observations on the raw data

> NaN columns:
LotFrontage: 259 nan, probably means there is no lot, so set nan to zero
Alley: 1369 nan, means no alley access

MasVnrType: 8 nan, may well just say they are 'None'
MasVnrArea: 8 nan, their 'MasVnrType' are null, i.e. it means 'None'

BsmtQual: 37 nan, means no basement
BsmtCond: 37 nan, means no basement
BsmtExposure: 38 nan, 37 is no basement, 1 is BsmtFinType1 and BsmtFinType2 'Unf' unfinished, but with BsmtQual and BsmtCond not nan
BsmtFinType1: 37 nan, means no basement
BsmtFinType2: 38 nan, means no basement, 1 has basement but nan; set it to 'Rec' (average)
set the nan to -1 or 'None'

Electrical: 1 nan, set to FuseA (average)
FireplaceQu: 690 nan, they have no fireplace (see Fireplaces column), set quality to -1

GarageType: 81 nan, no garage
GarageYrBlt, GarageFinish, GarageQual, GarageCond
set to -1 or 'None', and year built to the mean

PoolQC: 1543 nan, no pool (see PoolArea column)
Fence: 1179 nan, no fence
MiscFeature: 1406 nan, no misc features

> cardinalities: 
Neighborhood (25), Exterior1st (15) and Exterior2nd (16) have large (> 10) cardinalities, but they look important so let's keep them

> ordered categories (use ordinal encoding): 
'Street', 'Alley', 'LotShape', 'Utilities', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence'

> to do: 
drop Id column
train-validation split
take care of NaN as indicated above
ordinal encoding
one-hot encoding
scale all features

"""


#%% train-validation split

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, random_state = 0, train_size = 0.8)

np.save('X_train', X_train)
np.save('X_valid', X_valid)

#%% preprocessing

import preprocessing as pp

encoded_X_train = pp.preprocessing_fit_transform(X_train)
encoded_X_valid = pp.preprocessing_transform(X_valid)

# checked that all dtypes are int64 or float64
# encoded_X_train.apply(dtype,axis=0)
# encoded_X_valid.apply(dtype,axis=0)


encoded_X_train_scaled = pp.scaler_fit_transform(encoded_X_train)
encoded_X_valid_scaled = pp.scaler_transform(encoded_X_valid)

np.save('encoded_X_train', encoded_X_train)
np.save('encoded_X_train_scaled', encoded_X_train_scaled)
np.save('y_train', y_train)
np.save('encoded_X_valid', encoded_X_valid)
np.save('encoded_X_valid_scaled', encoded_X_valid_scaled)
np.save('y_valid', y_valid)

#%% model scores

from sklearn.metrics import mean_squared_log_error, r2_score

from sklearn.dummy import DummyRegressor
lm_dummy_mean = DummyRegressor(strategy = 'mean').fit(encoded_X_train, y_train)

print('Dummy Regression: training set R2 score is {:.3f}'.format(lm_dummy_mean.score(encoded_X_train, y_train)))
print('Dummy Regression: validation set R2 score is {:.3f}'.format(lm_dummy_mean.score(encoded_X_valid, y_valid)))
# Dummy Regression: training set R2 score is 0.000
# Dummy Regression: validation set R2 score is -0.000

print('Dummy Regression: training set root-mean-squared log error is {:.5f}'.format(mean_squared_log_error(y_train, lm_dummy_mean.predict(encoded_X_train))))
print('Dummy Regression: validation set root-mean-squared log error is {:.5f}'.format(mean_squared_log_error(y_valid, lm_dummy_mean.predict(encoded_X_valid))))
# Dummy Regression: training set root-mean-squared log error is 0.16816
# Dummy Regression: validation set root-mean-squared log error is 0.15754

#################################

from sklearn.linear_model import LinearRegression

linreg = LinearRegression(n_jobs = -1)
linreg.fit(encoded_X_train, y_train)

print('Linear Regression: training set R2 score is {:.3f}'.format(linreg.score(encoded_X_train, y_train)))
print('Linear Regression: validation set R2 score is {:.3f}'.format(linreg.score(encoded_X_valid, y_valid)))
# Linear Regression: training set R2 score is 0.931
# Linear Regression: validation set R2 score is 0.577

print('Linear Regression: training set root-mean-squared log error is {:.0f}'.format(mean_squared_log_error(y_train, linreg.predict(encoded_X_train))))
print('Linear Regression: validation set root-mean-squared log error is {:.0f}'.format(mean_squared_log_error(y_valid, linreg.predict(encoded_X_valid))))
# Linear Regression: training set root-mean-squared error is 20666
# Linear Regression: validation set root-mean-squared error is 54265


plt.figure()
plt.scatter(np.array(list(y_train.index)), np.array(list(y_train.values)))
plt.scatter(np.array(list(y_train.index)), linreg.predict(encoded_X_train))

plt.figure()
plt.scatter(np.array(list(y_valid.index)), np.array(list(y_valid.values)))
plt.scatter(np.array(list(y_valid.index)), linreg.predict(encoded_X_valid))

#################################

from sklearn.linear_model import Ridge
linridge = Ridge()

from sklearn.model_selection import GridSearchCV
grid_values_ridge = {'alpha': list(np.arange(2,5,0.5))}

linridge_mse = GridSearchCV(linridge, param_grid = grid_values_ridge, scoring = 'neg_root_mean_squared_error')
linridge_mse.fit(encoded_X_train_scaled, y_train)
print('Grid best parameter (min. neg_root_mean_squared_error): ', linridge_mse.best_params_)
# Grid best parameter (min. neg_root_mean_squared_error):  {'alpha': 3.0}

print('Ridge Regression with alpha = {}: training set R2 score is {:.3f}'.format(linridge_mse.best_params_['alpha'], r2_score(y_train, linridge_mse.predict(encoded_X_train_scaled))))
print('Ridge Regression with alpha = {}: validation set R2 score is {:.3f}'.format(linridge_mse.best_params_['alpha'], r2_score(y_valid, linridge_mse.predict(encoded_X_valid_scaled))))
# Ridge Regression with alpha = 3.0: training set R2 score is 0.913
# Ridge Regression with alpha = 3.0: validation set R2 score is 0.725

print('Ridge Regression with alpha = {}: training set root-mean-squared log error is {:.5f}'.format(linridge_mse.best_params_['alpha'], mean_squared_log_error(y_train, linridge_mse.predict(encoded_X_train_scaled))))
print('Ridge Regression with alpha = {}: validation set root-mean-squared log error is {:.5f}'.format(linridge_mse.best_params_['alpha'], mean_squared_log_error(y_valid, linridge_mse.predict(encoded_X_valid_scaled))))
# Ridge Regression with alpha = 3.0: training set root-mean-squared log error is 0.01573
# Ridge Regression with alpha = 3.0: validation set root-mean-squared log error is 0.03354

#################################

from sklearn.linear_model import Lasso
linlasso = Lasso(max_iter = 100000)

from sklearn.model_selection import GridSearchCV
grid_values_lasso = {'alpha': list(np.arange(50,80,1))}

linlasso_mse = GridSearchCV(linlasso, param_grid = grid_values_lasso, scoring = 'neg_root_mean_squared_error')
linlasso_mse.fit(encoded_X_train_scaled, y_train)
print('Grid best parameter (min. neg_root_mean_squared_error): ', linlasso_mse.best_params_)
# Grid best parameter (min. neg_root_mean_squared_error):  {'alpha': 69}

print('Lasso Regression with alpha = {}: training set R2 score is {:.3f}'.format(linlasso_mse.best_params_['alpha'], r2_score(y_train, linlasso_mse.predict(encoded_X_train_scaled))))
print('Lasso Regression with alpha = {}: validation set R2 score is {:.3f}'.format(linlasso_mse.best_params_['alpha'], r2_score(y_valid, linlasso_mse.predict(encoded_X_valid_scaled))))
# Lasso Regression with alpha = 69: training set R2 score is 0.918
# Lasso Regression with alpha = 69: validation set R2 score is 0.685

print('Lasso Regression with alpha = {}: training set root-mean-squared log error is {:.5f}'.format(linlasso_mse.best_params_['alpha'], mean_squared_log_error(y_train, linlasso_mse.predict(encoded_X_train_scaled))))
print('Lasso Regression with alpha = {}: validation set root-mean-squared log error is {:.5f}'.format(linlasso_mse.best_params_['alpha'], mean_squared_log_error(y_valid, linlasso_mse.predict(encoded_X_valid_scaled))))
# Lasso Regression with alpha = 69: training set root-mean-squared log error is 0.01691
# Lasso Regression with alpha = 69: validation set root-mean-squared log error is 0.03276

#################################

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
encoded_X_train_scaled_poly = poly.fit_transform(encoded_X_train_scaled)
encoded_X_valid_scaled_poly = poly.transform(encoded_X_valid_scaled)


linridge_poly = Ridge()
grid_values_ridge_poly = {'alpha': list(np.arange(60,80,1))}

linridge_mse_poly = GridSearchCV(linridge_poly, param_grid = grid_values_ridge_poly, scoring = 'neg_root_mean_squared_error')
linridge_mse_poly.fit(encoded_X_train_scaled_poly, y_train)
print('Grid best parameter (min. neg_root_mean_squared_error): ', linridge_mse_poly.best_params_)
# Grid best parameter (min. neg_root_mean_squared_error):  {'alpha': 64}

print('Ridge Polynomial (p = 2) Regression with alpha = {}: training set R2 score is {:.3f}'.format(linridge_mse_poly.best_params_['alpha'], r2_score(y_train, linridge_mse_poly.predict(encoded_X_train_scaled_poly))))
print('Ridge Polynomial (p = 2) Regression with alpha = {}: validation set R2 score is {:.3f}'.format(linridge_mse_poly.best_params_['alpha'], r2_score(y_valid, linridge_mse_poly.predict(encoded_X_valid_scaled_poly))))
# Ridge Polynomial (p = 2) Regression with alpha = 64: training set R2 score is 0.965
# Ridge Polynomial (p = 2) Regression with alpha = 64: validation set R2 score is 0.727

print('Ridge Polynomial (p = 2) Regression with alpha = {}: training set root-mean-squared log error is {:.5f}'.format(linridge_mse_poly.best_params_['alpha'], mean_squared_log_error(y_train, linridge_mse_poly.predict(encoded_X_train_scaled_poly))))
print('Ridge Polynomial (p = 2) Regression with alpha = {}: validation set root-mean-squared log error is {:.5f}'.format(linridge_mse_poly.best_params_['alpha'], mean_squared_log_error(y_valid, linridge_mse_poly.predict(encoded_X_valid_scaled_poly))))
# Ridge Polynomial (p = 2) Regression with alpha = 64: training set root-mean-squared log error is 0.00560
# Ridge Polynomial (p = 2) Regression with alpha = 64: validation set root-mean-squared log error is 0.02479

#################################

from sklearn.neighbors import KNeighborsRegressor
k = 7
knnreg = KNeighborsRegressor(n_neighbors = k).fit(encoded_X_train_scaled, y_train)

print('knn Regression with k = {}: training set R2 score is {:.3f}'.format(k, knnreg.score(encoded_X_train_scaled, y_train)))
print('knn Regression with k = {}: validation set R2 score is {:.3f}'.format(k, knnreg.score(encoded_X_valid_scaled, y_valid)))
# knn Regression with k = 7: training set R2 score is 0.799
# knn Regression with k = 7: validation set R2 score is 0.674

print('knn Regression with k = {}: training set root-mean-squared log error is {:.5f}'.format(k, mean_squared_log_error(y_train, knnreg.predict(encoded_X_train_scaled))))
print('knn Regression with k = {}: validation set root-mean-squared log error is {:.5f}'.format(k, mean_squared_log_error(y_valid, knnreg.predict(encoded_X_valid_scaled))))
# knn Regression with k = 7: training set root-mean-squared log error is 0.03079
# knn Regression with k = 7: validation set root-mean-squared log error is 0.03733

knnreg2 = KNeighborsRegressor(n_neighbors = k)
grid_values_knn = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7], 'p': [1, 2, 3]}

knnreg2_mse = GridSearchCV(knnreg2, param_grid = grid_values_knn, scoring = 'neg_root_mean_squared_error')
knnreg2_mse.fit(encoded_X_train_scaled, y_train)
print('Grid best parameter (min. neg_root_mean_squared_error): ', knnreg2_mse.best_params_)
# Grid best parameter (min. neg_root_mean_squared_error):  {'n_neighbors': 4, 'p': 1}

print('knn Regression with k = {}, p = {}: training set R2 score is {:.3f}'.format(knnreg2_mse.best_params_['n_neighbors'], knnreg2_mse.best_params_['p'], r2_score(y_train,knnreg2_mse.predict(encoded_X_train_scaled))))
print('knn Regression with k = {}, p = {}: validation set R2 score is {:.3f}'.format(knnreg2_mse.best_params_['n_neighbors'], knnreg2_mse.best_params_['p'], r2_score(y_valid,knnreg2_mse.predict(encoded_X_valid_scaled))))
# knn Regression with k = 4, p = 1: training set R2 score is 0.863
# knn Regression with k = 4, p = 1: validation set R2 score is 0.717

print('knn Regression with k = {}, p = {}: training set root-mean-squared log error is {:.5f}'.format(knnreg2_mse.best_params_['n_neighbors'], knnreg2_mse.best_params_['p'], mean_squared_log_error(y_train, knnreg2_mse.predict(encoded_X_train_scaled))))
print('knn Regression with k = {}, p = {}: validation set root-mean-squared log error is {:.5f}'.format(knnreg2_mse.best_params_['n_neighbors'], knnreg2_mse.best_params_['p'], mean_squared_log_error(y_valid, knnreg2_mse.predict(encoded_X_valid_scaled))))
# knn Regression with k = 4, p = 1: training set root-mean-squared log error is 0.02066
# knn Regression with k = 4, p = 1: validation set root-mean-squared log error is 0.03502

#################################

from xgboost import XGBRegressor

grid_values_XGBR = {'n_estimators': list(np.arange(800, 1000, 50)), 'learning_rate': [0.01], 'max_depth': [4, 5, 6, 7]}

XGBR_model = XGBRegressor(n_jobs = 4)
XGBR_model_mse = GridSearchCV(XGBR_model, param_grid = grid_values_XGBR, scoring = 'neg_root_mean_squared_error')
XGBR_model_mse.fit(encoded_X_train_scaled, y_train)
print('Grid best parameter (min. neg_root_mean_squared_error): ', XGBR_model_mse.best_params_)
# Grid best parameter (min. neg_root_mean_squared_error):  {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 950}


mean_squared_log_error(y_train, XGBR_model_mse.predict(encoded_X_train_scaled), squared = False)
mean_squared_log_error(y_valid, XGBR_model_mse.predict(encoded_X_valid_scaled), squared = False)

print('XGBRegressor with n_estimators = {}, learning_rate = {}, max_depth = {}: training set R2 score is {:.3f}'.format(XGBR_model_mse.best_params_['n_estimators'], XGBR_model_mse.best_params_['learning_rate'], XGBR_model_mse.best_params_['max_depth'], r2_score(y_train, XGBR_model_mse.predict(encoded_X_train_scaled))))
print('XGBRegressor with n_estimators = {}, learning_rate = {}, max_depth = {}: validation set R2 score is {:.3f}'.format(XGBR_model_mse.best_params_['n_estimators'], XGBR_model_mse.best_params_['learning_rate'], XGBR_model_mse.best_params_['max_depth'], r2_score(y_valid, XGBR_model_mse.predict(encoded_X_valid_scaled))))
# XGBRegressor with n_estimators = 950, learning_rate = 0.01, max_depth = 5: training set R2 score is 0.990
# XGBRegressor with n_estimators = 950, learning_rate = 0.01, max_depth = 5: validation set R2 score is 0.848

print('XGBRegressor with n_estimators = {}, learning_rate = {}, max_depth = {}: training set root-mean-squared log error is {:.5f}'.format(XGBR_model_mse.best_params_['n_estimators'], XGBR_model_mse.best_params_['learning_rate'], XGBR_model_mse.best_params_['max_depth'], mean_squared_log_error(y_train, XGBR_model_mse.predict(encoded_X_train_scaled))))
print('XGBRegressor with n_estimators = {}, learning_rate = {}, max_depth = {}: validation set root-mean-squared log error is {:.5f}'.format(XGBR_model_mse.best_params_['n_estimators'], XGBR_model_mse.best_params_['learning_rate'], XGBR_model_mse.best_params_['max_depth'], mean_squared_log_error(y_valid, XGBR_model_mse.predict(encoded_X_valid_scaled))))
# XGBRegressor with n_estimators = 950, learning_rate = 0.01, max_depth = 5: training set root-mean-squared log error is 0.00275
# XGBRegressor with n_estimators = 950, learning_rate = 0.01, max_depth = 5: validation set root-mean-squared log error is 0.01618

XGBR_best = XGBRegressor(n_estimators = 950, learning_rate = 0.01, max_depth = 5,  n_jobs = 4)
XGBR_best.fit(encoded_X_train_scaled, y_train)
mean_squared_log_error(y_valid, XGBR_best.predict(encoded_X_valid_scaled))
# 0.01618149366670845

from pickle import dump
# save the model
dump(XGBR_best, open('XGBR_best.pkl', 'wb'))



#%%

test_df_all = pd.read_csv(r'datasets\test.csv')

test_df_Id = test_df_all['Id']

test_df_all.shape
# (1459, 80)

encoded_X_test = pp.preprocessing_transform(test_df_all)

# checked that all dtypes are int64 or float64
# encoded_X_train.apply(dtype,axis=0)
# encoded_X_valid.apply(dtype,axis=0)

encoded_X_test_scaled = pp.scaler_transform(encoded_X_test)

y_test_pred_XGBR = pd.Series(XGBR_best.predict(encoded_X_test_scaled).squeeze(), index = test_df_Id, name = 'SalePrice')

y_test_pred_XGBR.to_csv('y_test_pred_XGBR.csv')

# 0.13381

y_test_pred_linridge_poly2 = pd.Series(linridge_mse_poly.predict(poly.transform(encoded_X_test_scaled)).squeeze(), index = test_df_Id, name = 'SalePrice')

y_test_pred_linridge_poly2.to_csv('y_test_pred_linridge_poly2.csv')

# 0.13703




