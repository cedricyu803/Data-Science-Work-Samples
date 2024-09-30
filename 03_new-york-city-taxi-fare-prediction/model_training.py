# -*- coding: utf-8 -*-
"""
Created on Wed Sep 08 19:00:00 2021

@author: Cedric Yu
"""

# %%
"""
#####################################

# https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/overview

# New York City Taxi Fare Prediction


#####################################

In this playground competition, hosted in partnership with Google Cloud and Coursera, you are tasked with predicting the fare amount (inclusive of tolls) for a taxi ride in New York City given the pickup and dropoff locations. While you can get a basic estimate based on just the distance between the two points, this will result in an RMSE of $5-$8, depending on the model used (see the starter code for an example of this approach in Kernels). Your challenge is to do better than this using Machine Learning techniques!

To learn how to handle large datasets with ease and solve this problem using TensorFlow, consider taking the Machine Learning with TensorFlow on Google Cloud Platform specialization on Coursera -- the taxi fare problem is one of several real-world problems that are used as case studies in the series of courses. To make this easier, head to Coursera.org/NEXTextended to claim this specialization for free for the first month!

#####################################
Datasets

File descriptions

# train.csv - Input features and target fare_amount values for the training set (about 55M rows).
# test.csv - Input features for the test set (about 10K rows). Your goal is to predict fare_amount for each row.
# sample_submission.csv - a sample submission file in the correct format (columns key and fare_amount). This file 'predicts' fare_amount to be $11.35 for all rows, which is the mean fare_amount from the training set.


# Data fields

# ID
# key - Unique string identifying each row in both the training and test sets. Comprised of pickup_datetime plus a unique integer, but this doesn't matter, it should just be used as a unique ID field. Required in your submission CSV. Not necessarily needed in the training set, but could be useful to simulate a 'submission file' while doing cross-validation within the training set.
# Features
# pickup_datetime - timestamp value indicating when the taxi ride started.
# pickup_longitude - float for longitude coordinate of where the taxi ride started.
# pickup_latitude - float for latitude coordinate of where the taxi ride started.
# dropoff_longitude - float for longitude coordinate of where the taxi ride ended.
# dropoff_latitude - float for latitude coordinate of where the taxi ride ended.
# passenger_count - integer indicating the number of passengers in the taxi ride.


# Target

# fare_amount - float dollar amount of the cost of the taxi ride. This value is only in the training set; this is what you are predicting in the test set and it is required in your submission CSV.

"""

# %%

"""
model training
"""

# %% Preamble

# Make the output look better
import lightgbm as lgb
from pickle import dump
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# default='warn' # ignores warning about dropping columns inplace
pd.options.mode.chained_assignment = None

# import dask
# import dask.dataframe as dd


os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\7_new-york-city-taxi-fare-prediction')


# %% load pre-processed datasets for model training
# !!!

train_cols = pd.read_csv('engineered_datasets/X_train_encoded4.csv',
                         nrows=0).columns.drop(['Unnamed: 0']).tolist()

# downcast datatypes to save RAM
dtypes_new = dict(zip(train_cols, ['float32', 'float32', 'float32', 'float32', bool, 'float32', 'float32', 'float32', 'float32', 'float32', bool, bool, bool, bool, bool, bool,
                  bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32']))

# X_train_encoded4 = dd.read_csv('engineered_datasets/X_train_encoded4.csv', usecols = train_cols, dtype = dtypes_new)
# X_valid_encoded4 = dd.read_csv('engineered_datasets/X_valid_encoded4.csv', usecols = train_cols, dtype = dtypes_new)

""" only load the first 20M rows of training set due to insufficient memory"""
X_train_encoded4 = pd.read_csv(
    'engineered_datasets/X_train_encoded4.csv', index_col=[0], dtype=dtypes_new, nrows=20000000)
X_valid_encoded4 = pd.read_csv(
    'engineered_datasets/X_valid_encoded4.csv', index_col=[0], dtype=dtypes_new)

# X_train_encoded4_scaled = pd.read_csv('engineered_datasets/2021-09-06_5/X_train_encoded4_scaled.csv', index_col = [0])
# X_valid_encoded4_scaled = pd.read_csv('engineered_datasets/2021-09-06_5/X_valid_encoded4_scaled.csv', index_col = [0])

y_cols = pd.read_csv('engineered_datasets/y_train.csv',
                     nrows=0).columns.drop(['Unnamed: 0']).tolist()

# y_train = dd.read_csv('engineered_datasets/y_train.csv', usecols = y_cols).squeeze()
# y_valid = dd.read_csv('engineered_datasets/y_valid.csv', usecols = y_cols).squeeze()

y_train = pd.read_csv('engineered_datasets/y_train.csv',
                      index_col=[0], nrows=20000000).squeeze()
y_valid = pd.read_csv('engineered_datasets/y_valid.csv',
                      index_col=[0]).squeeze()

gc.collect()

# %% model scores


#################################

"""# LinearRegression """

# from sklearn.linear_model import LinearRegression

# linreg = LinearRegression(n_jobs = -1)
# linreg.fit(X_train_encoded4_scaled, y_train)

# print('Linear Regression: training set R2 score is {:.3f}'.format(linreg.score(X_train_encoded4_scaled, y_train)))
# print('Linear Regression: validation set R2 score is {:.3f}'.format(linreg.score(X_valid_encoded4_scaled, y_valid)))
# # Linear Regression: training set R2 score is 0.610
# # Linear Regression: validation set R2 score is 0.612

# print('Linear Regression: training set root-mean-squared log error is {:.3f}'.format(mean_squared_error(y_train, linreg.predict(X_train_encoded4_scaled), squared = False)))
# print('Linear Regression: validation set root-mean-squared log error is {:.3f}'.format(mean_squared_error(y_valid, linreg.predict(X_valid_encoded4_scaled), squared = False)))


# plt.figure()
# plt.scatter(np.array(list(y_train.index)), np.array(list(y_train.values)))
# plt.scatter(np.array(list(y_train.index)), linreg.predict(X_train_encoded4_scaled))

# plt.figure()
# plt.scatter(np.array(list(y_valid.index)), np.array(list(y_valid.values)))
# plt.scatter(np.array(list(y_valid.index)), linreg.predict(X_valid_encoded4_scaled))

#################################

"""# Ridge """
# from sklearn.linear_model import Ridge
# linridge = Ridge()

# from sklearn.model_selection import GridSearchCV
# grid_values_ridge = {'alpha': list(np.arange(2,5,0.5))}

# linridge_mse = GridSearchCV(linridge, param_grid = grid_values_ridge, scoring = 'neg_root_mean_squared_error')
# linridge_mse.fit(X_train_encoded4_scaled, y_train)
# print('Grid best parameter (min. neg_root_mean_squared_error): ', linridge_mse.best_params_)
# # Grid best parameter (min. neg_root_mean_squared_error):  {'alpha': 2.0}

# print('Ridge Regression with alpha = {}: training set R2 score is {:.3f}'.format(linridge_mse.best_params_['alpha'], linridge_mse.score(X_train_encoded4_scaled, y_train)))
# print('Ridge Regression with alpha = {}: validation set R2 score is {:.3f}'.format(linridge_mse.best_params_['alpha'], linridge_mse.score(X_valid_encoded4_scaled, y_valid)))
# # Ridge Regression with alpha = 2.0: training set R2 score is -417.271
# # Ridge Regression with alpha = 2.0: validation set R2 score is -416.735

# print('Ridge Regression with alpha = {}: training set root-mean-squared log error is {:.3f}'.format(linridge_mse.best_params_['alpha'], np.sqrt(mean_squared_log_error(y_train, linridge_mse.predict(X_train_encoded4_scaled)))))
# print('Ridge Regression with alpha = {}: validation set root-mean-squared log error is {:.3f}'.format(linridge_mse.best_params_['alpha'], np.sqrt(mean_squared_log_error(y_valid, linridge_mse.predict(X_valid_encoded4_scaled)))))
# # Ridge Regression with alpha = 2.0: training set root-mean-squared log error is 0.511
# # Ridge Regression with alpha = 2.0: validation set root-mean-squared log error is 0.511

#################################

"""# Lasso """
# from sklearn.linear_model import Lasso
# linlasso = Lasso(max_iter = 100000)

# from sklearn.model_selection import GridSearchCV
# grid_values_lasso = {'alpha': list(np.arange(50,80,1))}

# linlasso_mse = GridSearchCV(linlasso, param_grid = grid_values_lasso, scoring = 'neg_root_mean_squared_error')
# linlasso_mse.fit(encoded_X_train_scaled, y_train)
# print('Grid best parameter (min. neg_root_mean_squared_error): ', linlasso_mse.best_params_)
# # Grid best parameter (min. neg_root_mean_squared_error):  {'alpha': 68}

# print('Lasso Regression with alpha = {}: training set R2 score is {:.3f}'.format(linlasso_mse.best_params_['alpha'], linlasso_mse.score(encoded_X_train_scaled, y_train)))
# print('Lasso Regression with alpha = {}: validation set R2 score is {:.3f}'.format(linlasso_mse.best_params_['alpha'], linlasso_mse.score(encoded_X_valid_scaled, y_valid)))
# # Lasso Regression with alpha = 68: training set R2 score is -22411.582
# # Lasso Regression with alpha = 68: validation set R2 score is -46851.601

# print('Lasso Regression with alpha = {}: training set root-mean-squared log error is {:.0f}'.format(linlasso_mse.best_params_['alpha'], mean_squared_log_error(y_train, linlasso_mse.predict(encoded_X_train_scaled), squared = False)))
# print('Lasso Regression with alpha = {}: validation set root-mean-squared log error is {:.0f}'.format(linlasso_mse.best_params_['alpha'], mean_squared_log_error(y_valid, linlasso_mse.predict(encoded_X_valid_scaled), squared = False)))
# # Lasso Regression with alpha = 68: training set root-mean-squared error is 22412
# # Lasso Regression with alpha = 68: validation set root-mean-squared error is 46852

#################################

"""# Polynomial """

# from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(degree=3)
# encoded_X_train_scaled_poly = poly.fit_transform(X_train_encoded4_scaled)
# encoded_X_valid_scaled_poly = poly.transform(X_valid_encoded4_scaled)

# from sklearn.linear_model import Ridge
# linridge_poly = Ridge()
# grid_values_ridge_poly = {'alpha': list(np.arange(0,80,10))}

# from sklearn.model_selection import GridSearchCV
# linridge_mse_poly = GridSearchCV(linridge_poly, param_grid = grid_values_ridge_poly, scoring = 'neg_mean_squared_log_error')
# linridge_mse_poly.fit(encoded_X_train_scaled_poly, y_train)
# print('Grid best parameter (min. mean_squared_log_error): ', linridge_mse_poly.best_params_)
# #

# print('Ridge Polynomial (p = 3) Regression with alpha = {}: training set R2 score is {:.3f}'.format(linridge_mse_poly.best_params_['alpha'], linridge_mse_poly.score(encoded_X_train_scaled_poly, y_train)))
# print('Ridge Polynomial (p = 3) Regression with alpha = {}: validation set R2 score is {:.3f}'.format(linridge_mse_poly.best_params_['alpha'], linridge_mse_poly.score(encoded_X_valid_scaled_poly, y_valid)))

# print('Ridge Polynomial (p = 3) Regression with alpha = {}: training set root-mean-squared log error is {:.3f}'.format(linridge_mse_poly.best_params_['alpha'], np.sqrt(mean_squared_error(y_train, linridge_mse_poly.predict(encoded_X_train_scaled_poly), squared = False))))
# print('Ridge Polynomial (p = 3) Regression with alpha = {}: validation set root-mean-squared log error is {:.3f}'.format(linridge_mse_poly.best_params_['alpha'], np.sqrt(mean_squared_error(y_valid, linridge_mse_poly.predict(encoded_X_valid_scaled_poly), squared = False))))


#################################

"""# knn """

# from sklearn.neighbors import KNeighborsRegressor
# k = 5
# knnreg = KNeighborsRegressor(n_neighbors = k, n_jobs = 6).fit(X_train_encoded4_scaled, y_train)
# # took half an hour to train

# print('knn Regression with k = {}: training set R2 score is {:.3f}'.format(k, knnreg.score(X_train_encoded4_scaled, y_train)))
# print('knn Regression with k = {}: validation set R2 score is {:.3f}'.format(k, knnreg.score(X_valid_encoded4_scaled, y_valid)))
# # knn Regression with k = 5: training set R2 score is 0.751
# # knn Regression with k = 5: validation set R2 score is 0.624

# print('knn Regression with k = {}: training set root-mean-squared log error is {:.3f}'.format(k, np.sqrt(mean_squared_log_error(y_train, knnreg.predict(X_train_encoded4_scaled)))))
# print('knn Regression with k = {}: validation set root-mean-squared log error is {:.3f}'.format(k, np.sqrt(mean_squared_log_error(y_valid, knnreg.predict(X_valid_encoded4_scaled)))))
# # knn Regression with k = 5: training set root-mean-squared log error is 0.367
# # knn Regression with k = 5: validation set root-mean-squared log error is 0.449
# # getting better; overfitting


# from pickle import dump
# # save the model
# dump(knnreg, open('knnreg5.pkl', 'wb'))


# this will take forever...
# knnreg2 = KNeighborsRegressor()
# grid_values_knn = {'n_neighbors': [3, 5, 7], 'p': [1]}

# from sklearn.model_selection import GridSearchCV
# knnreg2_rmsle = GridSearchCV(knnreg2, param_grid = grid_values_knn, scoring = 'neg_mean_squared_log_error', verbose = 1)
# knnreg2_rmsle.fit(X_train_encoded4_scaled, y_train)
# print('Grid best parameter (min. mean_squared_log_error): ', knnreg2_rmsle.best_params_)
# # Grid best parameter (min. neg_root_mean_squared_error):  {'n_neighbors': 4, 'p': 1}

# print('knn Regression with k = {}, p = {}: training set R2 score is {:.3f}'.format(knnreg2_rmsle.best_params_['n_neighbors'], knnreg2_rmsle.best_params_['p'], knnreg2_rmsle.score(X_train_encoded4_scaled, y_train)))
# print('knn Regression with k = {}, p = {}: validation set R2 score is {:.3f}'.format(knnreg2_rmsle.best_params_['n_neighbors'], knnreg2_rmsle.best_params_['p'], knnreg2_rmsle.score(X_valid_encoded4_scaled, y_valid)))
# # knn Regression with k = 4, p = 1: training set R2 score is -29032.238
# # knn Regression with k = 4, p = 1: validation set R2 score is -44316.140

# print('knn Regression with k = {}, p = {}: training set root-mean-squared log error is {:.3f}'.format(knnreg2_rmsle.best_params_['n_neighbors'], knnreg2_rmsle.best_params_['p'], np.sqrt(mean_squared_log_error(y_train, knnreg2_rmsle.predict(X_train_encoded4_scaled)))))
# print('knn Regression with k = {}, p = {}: validation set root-mean-squared log error is {:.3f}'.format(knnreg2_rmsle.best_params_['n_neighbors'], knnreg2_rmsle.best_params_['p'], np.sqrt(mean_squared_log_error(y_valid, knnreg2_rmsle.predict(X_valid_encoded4_scaled)))))
# knn Regression with k = 4, p = 1: training set root-mean-squared error is 29032
# knn Regression with k = 4, p = 1: validation set root-mean-squared error is 44316

#################################

# !!!
"""# XGBRegressor """


# use at most 5M training instances for fitting
XGBR_model = XGBRegressor(eval_metric="rmse",
                          learning_rate=0.05,
                          max_depth=8,
                          n_estimators=100,
                          reg_lambda=0.7,
                          n_jobs=6)
XGBR_model.fit(X_train_encoded4, y_train, eval_set=[
               (X_train_encoded4, y_train), (X_valid_encoded4, y_valid)], early_stopping_rounds=40)
# XGBR_model.fit(X_train_encoded4_scaled, y_train, eval_set = [(X_train_encoded4_scaled, y_train), (X_valid_encoded4_scaled, y_valid)])
# del XGBR_model


# XGBR_model = XGBRegressor(eval_metric="rmsle", n_jobs = 8)
# XGBR_model.fit(X_train_encoded4_scaled, y_train, eval_set = [(X_train_encoded4_scaled, y_train), (X_valid_encoded4_scaled, y_valid)])


# print(mean_squared_error(y_train, XGBR_model.predict(X_train_encoded4_scaled), squared = False))
# print(mean_squared_error(y_valid, XGBR_model.predict(X_valid_encoded4_scaled), squared = False))


XGBR_feature_importances = pd.Series(
    XGBR_model.feature_importances_, index=X_train_encoded4.columns).sort_values(ascending=False)
# euclidean_distance_miles        0.719793
# dropoff_county_not_in_NYC       0.049462
# dropoff_airport                 0.040797
# dropoff_pickup_same_county      0.039250
# year_mean_encoded               0.019108
# dropoff_zipcode_mean_encoded    0.017932
# pickup_county_not_in_NYC        0.011769
# long_displacement               0.007071
# pickup_zipcode_mean_encoded     0.007070
# pickup_county_Queens            0.006466
# cos_direction                   0.005012
# dropoff_county_New York         0.004920
# pickup_airport                  0.004747
# pickup_county_New York          0.004522
# hour_mean_encoded               0.004261
# sin_direction                   0.004181
# dropoff_longitude               0.003957
# month_mean_encoded              0.003928
# lat_displacement                0.003871
# pickup_latitude                 0.003580
# dropoff_latitude                0.003541
# dropoff_county_Richmond         0.003384
# passenger_count_5               0.003369
# pickup_longitude                0.003182
# dropoff_county_Queens           0.003081
# passenger_count_1               0.002905
# dropoff_county_Kings            0.002303
# dropoff_county_Bronx            0.002259
# day_mean_encoded                0.002169
# weekday_freq_encoded            0.002147
# passenger_count_3               0.002073
# passenger_count_6               0.001911
# passenger_count_2               0.001695
# passenger_count_4               0.001260
# pickup_county_Kings             0.000972
# daylight_saving                 0.000877
# pickup_county_Bronx             0.000618
# pickup_county_Richmond          0.000557
# dtype: float32

# XGBR_feature_importances.plot.barh()
# plt.title('XGBR')
# plt.savefig('XGBR_feature_importances', dpi = 300)

# dask: not used
# from dask.distributed import Client
# client = Client()
# import xgboost

# X_train_encoded4 = X_train_encoded4.repartition(npartitions = 1)
# X_valid_encoded4 = X_valid_encoded4.repartition(npartitions = 1)
# y_train = y_train.repartition(npartitions = 1)
# y_valid = y_valid.repartition(npartitions = 1)

# dtrain = xgboost.dask.DaskDMatrix(client, X_train_encoded4, y_train)
# dvalid = xgboost.dask.DaskDMatrix(client, X_valid_encoded4, y_valid)


# dask_XGBR_model0 = xgboost.dask.train(client,
#         {"verbosity": 2, "objective": "reg:squarederror"},
#         dtrain,
#         num_boost_round=10,
#         evals=[(dtrain, "train")],
#     )

# gc.collect()

# dask_XGBR_model = xgboost.dask.train(client, params, dtrain, num_boost_round = 10, evals = [(dtrain, 'train'), (dvalid, 'valid')], early_stopping_round = 5)

# y_valid = y_valid.repartition(npartitions = 1)
# X_valid_encoded4 = X_valid_encoded4.repartition(npartitions = 1)


# dask_XGBR_model = xgb.dask.train(client, params, dtrain)


# xgboost.dask.predict(client, dask_XGBR_model0, X_valid_encoded4).compute()

# dask_XGBR_model = dxgb.XGBRegressor(client, eval_metric = "rmse",
#                           learning_rate = 0.05,
#                           max_depth = 8,
#                           n_estimators = 100,
#                           reg_lambda = 0.7)

# dask_XGBR_model.fit(X_train_encoded4, y_train, evals = [(X_train_encoded4, y_train), (X_valid_encoded4, y_valid)])

"""# RandomizedSearchCV """
# from sklearn.model_selection import RandomizedSearchCV

# grid_values_XGBR = {'n_estimators': list(np.arange(100, 1100, 100)), 'learning_rate': [0.001, 0.01, 0.1], 'max_depth': [3, 4, 5, 6, 7]}

# fit_params={"early_stopping_rounds": 50,
#             "eval_metric" : "rmsle",
#             "eval_set" : [[X_valid_encoded4_scaled, y_valid]]}

# XGBR_model2 = XGBRegressor(eval_metric="rmsle", n_jobs = 8)
# XGBR_model_rmsle = RandomizedSearchCV(XGBR_model2, param_distributions = grid_values_XGBR, scoring = 'neg_mean_squared_log_error', cv = 5, verbose = 1, n_iter = 4)
# XGBR_model_rmsle.fit(X_train_encoded4_scaled, y_train, **fit_params)
# print('Grid best parameter (min. mean_squared_log_error): ', XGBR_model_rmsle.best_params_)


# np.sqrt(mean_squared_log_error(y_train, XGBR_model_rmsle.predict(X_train_encoded4_scaled)))
# np.sqrt(mean_squared_log_error(y_valid, XGBR_model_rmsle.predict(X_valid_encoded4_scaled)))


# print('XGBRegressor with n_estimators = {}, learning_rate = {}, max_depth = {}: training set root-mean-squared log error is {:.0f}'.format(XGBR_model_rmsle.best_params_['n_estimators'], XGBR_model_rmsle.best_params_['learning_rate'], XGBR_model_rmsle.best_params_['max_depth'], np.sqrt(mean_squared_log_error(y_train, XGBR_model_rmsle.predict(X_train_encoded4_scaled)))))
# print('XGBRegressor with n_estimators = {}, learning_rate = {}, max_depth = {}: validation set root mean_squared_log_error is {:.0f}'.format(XGBR_model_rmsle.best_params_['n_estimators'], XGBR_model_rmsle.best_params_['learning_rate'], XGBR_model_rmsle.best_params_['max_depth'], np.sqrt(mean_squared_log_error(y_valid, XGBR_model_rmsle.predict(X_valid_encoded4_scaled)))))


# pd.Series(XGBR_model_rmsle.feature_importances_, index = X_train_encoded4_scaled.columns).sort_values(ascending = False)


# XGBR_best = XGBRegressor(n_estimators = 950, learning_rate = 0.01, max_depth = 5,  eval_metric="rmsle", n_jobs = 8)
# XGBR_best.fit(X_train_encoded4_scaled, y_train)
# np.sqrt(mean_squared_log_error(y_valid, XGBR_best.predict(X_valid_encoded4_scaled)))

# from pickle import dump
# # save the model
# dump(XGBR_model, open('XGBR_model.pkl', 'wb'))


#################################

"""# RandomForestRegressor """

# from sklearn.ensemble import RandomForestRegressor
# RFR = RandomForestRegressor(n_estimators = 400, random_state = 0, n_jobs = -1, verbose = 1)
# RFR.fit(X_train_encoded4_scaled, y_train)


# from sklearn.metrics import mean_squared_error
# print(mean_squared_error(y_train, RFR.predict(X_train_encoded4_scaled), squared = False))
# print(mean_squared_error(y_valid, RFR.predict(X_valid_encoded4_scaled), squared = False))


# pd.Series(RFR.feature_importances_, index = X_train_encoded4_scaled.columns).sort_values(ascending = False).plot.barh()


# from pickle import dump
# # save the model
# dump(RFR, open('RFR_model.pkl', 'wb'))


# from sklearn.model_selection import RandomizedSearchCV

# # Number of trees in random forest
# n_estimators = np.arange(100, 1050, 50)
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [10, 20, 40, 60, 80, 100, None]
# # Minimum number of samples required to split a node
# min_samples_split = [2, 8, 16, 128, 1024, 4096]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 4, 128, 512, 1024, 4096]

# # Create the random grid
# random_grid_RFR = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf}


# RFR_model2 = RandomForestRegressor(random_state = 0, n_jobs = -1)
# RFR_model_rmsle = RandomizedSearchCV(RFR_model2, param_distributions = random_grid_RFR, scoring = 'neg_mean_squared_log_error', cv = 5, verbose = 2, n_iter = 10)
# RFR_model_rmsle.fit(X_train_encoded4_scaled, y_train)
# # each CV (!) took about 15 mins
# print('Grid best parameter (min. mean_squared_log_error): ', RFR_model_rmsle.best_params_)


# RFR_best = RandomForestRegressor(n_estimators = 650, max_depth = 60, max_features= 'sqrt', min_samples_leaf = 4, min_samples_split = 128, random_state = 0, n_jobs = -1)
# RFR_best.fit(X_train_encoded4_scaled, y_train)

# print(np.sqrt(mean_squared_log_error(y_train, RFR_best.predict(X_train_encoded4_scaled))))
# print(np.sqrt(mean_squared_log_error(y_valid, RFR_best.predict(X_valid_encoded4_scaled))))


# pd.Series(RFR_best.feature_importances_, index = X_train_encoded4_scaled.columns).sort_values(ascending = False)


# plt.figure()
# plt.scatter(np.array(list(y_train.index)), np.array(list(y_train.values)))
# plt.scatter(np.array(list(y_train.index)), RFR_best.predict(X_train_encoded4_scaled))

# plt.figure()
# plt.scatter(np.array(list(y_valid.index)), np.array(list(y_valid.values)))
# plt.scatter(np.array(list(y_valid.index)), RFR_best.predict(X_valid_encoded4_scaled))

# from pickle import dump
# # save the model
# dump(RFR_best, open('RFR_best.pkl', 'wb'))


# from pickle import load
# RFR_best = load(open('RFR_best.pkl', 'rb'))

# my_pred = pd.Series(RFR_best.predict(encoded_X_test_scaled).clip(0,20), name = 'item_cnt_month', index = test_df_all['ID'])
# # test score 1.69278
# my_pred.to_csv('my_submission_RFR_best.csv')


#################################

"""# lightGBM"""


# training set size: at most 20M
LGBMreg = lgb.LGBMRegressor(boosting_type='gbdt',
                            learning_rate=0.02,
                            num_leaves=800,
                            n_estimators=500,
                            num_iterations=5000,
                            max_bin=500,
                            feature_fraction=0.7,
                            bagging_fraction=0.7,
                            lambda_l2=0.5,
                            max_depth=25,
                            silent=False
                            )

LGBMreg.fit(X_train_encoded4, y_train,
            eval_set=[(X_train_encoded4, y_train),
                      (X_valid_encoded4, y_valid)],
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbose=True)


# LGBMreg.fit(X_train_encoded4_scaled, y_train,
#             eval_set=[(X_train_encoded4_scaled, y_train), (X_valid_encoded4_scaled, y_valid)],
#             # eval_metric = rmsle,
#             eval_metric = 'rmse',
#             early_stopping_rounds = 100,
#             verbose = True)


# print(mean_squared_error(y_train, LGBMreg.predict(X_train_encoded4_scaled), squared = False))
# print(mean_squared_error(y_valid, LGBMreg.predict(X_valid_encoded4_scaled), squared = False))

LGBMreg_feature_importances = pd.Series(
    LGBMreg.feature_importances_, index=X_train_encoded4.columns).sort_values(ascending=False)
LGBMreg_feature_importances = LGBMreg_feature_importances / \
    LGBMreg_feature_importances.max()

fig = plt.figure('feature_importances', figsize=(10, 6), dpi=200)
sns.barplot(y=LGBMreg_feature_importances.iloc[:18].index,
            x=LGBMreg_feature_importances.iloc[:18].values, color='skyblue')
# LGBMreg_feature_importances.plot.barh()
ax = plt.gca()
ax.set_xlabel('Feature importance')
ax.set_ylabel(None)
ax.set_yticklabels(ax.get_yticklabels(), fontsize='small')
ax.set_title('lightGBM Regressor')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(0, 1)
fig.tight_layout()
# plt.savefig('plots/lightgbm_feature_importances20M.png', dpi = 200)

# save the model
dump(LGBMreg, open('LGBMreg.pkl', 'wb'))
