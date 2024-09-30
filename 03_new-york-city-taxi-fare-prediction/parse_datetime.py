# -*- coding: utf-8 -*-
"""
Created on Tue Sep 07 14:30:00 2021

@author: Cedric Yu
"""

# import matplotlib.pyplot as plt
# import math
# import re


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
parse datetime of the two original datasets, save as for future use, to save loading time.
"""

# %% Preamble

# Make the output look better
from datetime import datetime
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# default='warn' # ignores warning about dropping columns inplace
pd.options.mode.chained_assignment = None
# import re

os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\7_new-york-city-taxi-fare-prediction')

# %% load original train.csv dataset

# get training set column names
train_cols = pd.read_csv(r'datasets\train.csv', nrows=0).columns.tolist()

# parse datetime columns

"""# UTC timezone"""


def parser(s):
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S %Z')


"""# downcast datatypes to save RAM"""
dtypes = dict(zip(train_cols, [
              str, 'float32', str, 'float32', 'float32', 'float32', 'float32', 'uint8']))
parse_dates = [2]

# import original training dataset
train_df_raw = pd.read_csv(r'datasets\train.csv', low_memory=True,
                           dtype=dtypes, parse_dates=parse_dates, date_parser=parser)

# train_df_raw.shape

train_df = train_df_raw.copy()

# %% load original test.csv dataset

# get test set column names
test_cols = pd.read_csv(r'datasets\test.csv', nrows=0).columns.tolist()

# parse datetime columns

"""# UTC timezone"""


def parser(s):
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S %Z')


"""# downcast datatypes to save RAM"""
dtypes_test = dict(
    zip(test_cols, [str, str, 'float32', 'float32', 'float32', 'float32', 'uint8']))
parse_dates_test = [1]

# import original training dataset
test_df_raw = pd.read_csv(r'datasets\test.csv', low_memory=True,
                          dtype=dtypes_test, parse_dates=parse_dates_test, date_parser=parser)

test_df = test_df_raw.copy()

# %% functions for extracting datetime features

"""# year, month, weekday, day, hour, daylight_saving"""


def get_year(row):
    return row.year


def get_month(row):
    return row.month


def get_weekday(row):
    return row.weekday()


def get_day(row):
    return row.day


def get_hour(row):
    return row.hour


def daylight_saving(row):
    return row.tzname()


"""# convert UTC to Eastern Time (ET)"""


def UTC_to_ET(row):
    return row.tz_localize('UTC').tz_convert('America/New_York')

# %% extract datetime features


train_df['year'] = train_df['pickup_datetime'].apply(get_year)
train_df['month'] = train_df['pickup_datetime'].apply(get_month)
train_df['weekday'] = train_df['pickup_datetime'].apply(get_weekday)
train_df['day'] = train_df['pickup_datetime'].apply(get_day)
train_df['hour'] = train_df['pickup_datetime'].apply(get_hour)
train_df['pickup_datetime'] = train_df['pickup_datetime'].apply(UTC_to_ET)
train_df['daylight_saving'] = train_df['pickup_datetime'].apply(
    daylight_saving)

train_df['daylight_saving'] = train_df['daylight_saving'].replace({
                                                                  'EDT': 1, 'EST': 0})
train_df['daylight_saving'] = train_df['daylight_saving'].astype(bool)

# drop datetime column to save ram and load time
train_df = train_df.drop('pickup_datetime', axis=1)

train_df.to_csv('engineered_datasets/train_df_datetime.csv')


"""# convert UTC to ET"""

test_df['pickup_datetime'] = test_df['pickup_datetime'].apply(UTC_to_ET)

"""# extract year, month, weekday, day, hour, daylight_saving"""

test_df['year'] = test_df['pickup_datetime'].apply(get_year)
test_df['month'] = test_df['pickup_datetime'].apply(get_month)
test_df['weekday'] = test_df['pickup_datetime'].apply(get_weekday)
test_df['day'] = test_df['pickup_datetime'].apply(get_day)
test_df['hour'] = test_df['pickup_datetime'].apply(get_hour)
test_df['daylight_saving'] = test_df['pickup_datetime'].apply(daylight_saving)
test_df['daylight_saving'] = test_df['daylight_saving'].replace(
    {'EDT': 1, 'EST': 0})
test_df['daylight_saving'] = test_df['daylight_saving'].astype(bool)

# drop datetime column to save ram and load time
test_df = test_df.drop('pickup_datetime', axis=1)


test_df.to_csv('engineered_datasets/test_df_datetime.csv')
