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
from pickle import load
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# default='warn' # ignores warning about dropping columns inplace
pd.options.mode.chained_assignment = None

# import dask
# import dask.dataframe as dd


os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\7_new-york-city-taxi-fare-prediction')

# %% test dataset


# %% predict

test_df_cols = pd.read_csv(
    'engineered_datasets/test_df_datetime.csv', nrows=0).columns
test_df_cols = test_df_cols.drop(['Unnamed: 0']).tolist()

X_test_encoded4 = pd.read_csv(
    r'engineered_datasets\test_df_datetime.csv', usecols=test_df_cols)

X_test_id = X_test_encoded4['key'].copy()
X_test_encoded4 = X_test_encoded4.drop('key', axis=1)

LGBMreg = load(open('LGBMreg.pkl', 'rb'))

y_test_predict = pd.Series(LGBMreg.predict(
    X_test_encoded4), index=X_test_id, name='fare_amount')

y_test_predict.to_csv('predictions/y_test_predict_LGBMreg.csv')


fig = plt.figure('fare_amount_test_hist', dpi=150)
y_test_predict.to_frame()['fare_amount'].hist(
    bins=100, grid=False, ax=plt.gca(), color='tomato')
plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Fare amount in USD')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0.1,)
# plt.savefig('plots/fare_amount_pred_hist.png', dpi = 150)
