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


# %% Preamble

# Make the output look better
from sklearn.neighbors import KDTree
from simpledbf import Dbf5
import pyproj
from shapely.geometry import Point, Polygon
import geopandas  as gpd
import descartes
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


# !!!
# %% load datetime-processed training set

# get training set column names
train_cols_new = pd.read_csv(
    'engineered_datasets/train_df_datetime.csv', nrows=0).columns
train_cols_new = train_cols_new.drop('Unnamed: 0').to_list()

# downcast datatypes to save RAM
dtypes_new = dict(zip(train_cols_new, [str, 'float32', 'float32', 'float32', 'float32',
                  'float32', 'uint8', 'uint16', 'uint8', 'uint8', 'uint8', 'uint8', bool]))

# import datetime-processed training dataset
train_df_raw = pd.read_csv('engineered_datasets/train_df_datetime.csv',
                           low_memory=True, usecols=train_cols_new, dtype=dtypes_new)


# train_df_raw.shape

train_df = train_df_raw.copy()

test_df = pd.read_csv('engineered_datasets/test_df_datetime.csv')
test_df.drop(['Unnamed: 0'], axis=1, inplace=True)

# %% find coordinate nodes for distance calculations

# !!!
# take 10000 samples, see if driving distance ~ euclidean distance

# too slow; not used
"""
use osmnx and networkx to find driving distance
we find nearst node for all train_df_raw coordinates once and for all, save the dataset for future use
"""

# import networkx as nx
# import osmnx as ox
# # G = ox.graph_from_bbox(lat_max, lat_min, long_max, long_min, network_type='drive')
# G = ox.io.load_graphml("nyc_driving_network.graphml")

# train_df_raw['orig'] = ox.distance.nearest_nodes(G, train_df_raw['pickup_longitude'], train_df_raw['pickup_latitude'])
# train_df_raw['dest'] = ox.distance.nearest_nodes(G, train_df_raw['dropoff_longitude'], train_df_raw['dropoff_latitude'])

# train_df_raw.to_csv('engineered_datasets/test_df_datetime_coord_node.csv')

# def driving_distance(row):
#     try:
#         row['driving_distance'] = nx.shortest_path_length(G, row['orig'], row['dest'], weight='length')
#     except : # in case path is not found
#         row['driving_distance'] = np.nan
#         pass

#     return row

# train_df_raw = train_df_raw.apply(driving_distance, axis = 1)


# %% general observations

train_df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 55423856 entries, 0 to 55423855
# Data columns (total 13 columns):
#  #   Column             Dtype
# ---  ------             -----
#  0   key                object
#  1   fare_amount        float32
#  2   pickup_longitude   float32
#  3   pickup_latitude    float32
#  4   dropoff_longitude  float32
#  5   dropoff_latitude   float32
#  6   passenger_count    uint8
#  7   year               uint16
#  8   month              uint8
#  9   weekday            uint8
#  10  day                uint8
#  11  hour               uint8
#  12  daylight_saving    bool
# dtypes: bool(1), float32(5), object(1), uint16(1), uint8(5)
# memory usage: 1.9+ GB

train_df.describe()
#         fare_amount  pickup_longitude  pickup_latitude  dropoff_longitude  \
# count  5.542386e+07      5.542386e+07     5.542386e+07       5.542348e+07
# mean   8.077921e+00     -3.875290e+01     1.937328e+01      -3.875261e+01
# std    2.055127e+01      2.556202e+01     1.414234e+01       2.553839e+01
# min   -3.000000e+02     -3.442060e+03    -3.492264e+03      -3.442025e+03
# 25%    6.000000e+00     -7.399207e+01     4.073493e+01      -7.399140e+01
# 50%    8.500000e+00     -7.398180e+01     4.075265e+01      -7.398015e+01
# 75%    1.250000e+01     -7.396708e+01     4.076713e+01      -7.396368e+01
# max    9.396336e+04      3.457626e+03     3.408790e+03       3.457622e+03

#        dropoff_latitude  passenger_count          year         month  \
# count      5.542348e+07     5.542386e+07  5.542386e+07  5.542386e+07
# mean       1.937341e+01     1.685380e+00  2.011739e+03  6.270155e+00
# std        1.412191e+01     1.327664e+00  1.860034e+00  3.436072e+00
# min       -3.547887e+03     0.000000e+00  2.008000e+03  1.000000e+00
# 25%        4.073403e+01     1.000000e+00  2.010000e+03  3.000000e+00
# 50%        4.075316e+01     1.000000e+00  2.012000e+03  6.000000e+00
# 75%        4.076810e+01     2.000000e+00  2.013000e+03  9.000000e+00
# max        3.537133e+03     2.080000e+02  2.015000e+03  1.200000e+01

#             weekday           day          hour
# count  5.542386e+07  5.542386e+07  5.542386e+07
# mean   2.984923e+00  1.572758e+01  1.182400e+01
# std    1.905105e+00  8.683734e+00  5.836336e+00
# min    0.000000e+00  1.000000e+00  0.000000e+00
# 25%    1.000000e+00  8.000000e+00  7.000000e+00
# 50%    3.000000e+00  1.600000e+01  1.200000e+01
# 75%    5.000000e+00  2.300000e+01  1.700000e+01
# max    6.000000e+00  3.100000e+01  2.300000e+01


""" # has NaN"""
train_df.isnull().sum()
# key                    0
# fare_amount            0
# pickup_longitude       0
# pickup_latitude        0
# dropoff_longitude    376
# dropoff_latitude     376
# passenger_count        0
# year                   0
# month                  0
# weekday                0
# day                    0
# hour                   0
# daylight_saving        0
# dtype: int64

train_df.head(10)


train_df['passenger_count'].value_counts()
# 1      38337524
# 2       8175243
# 5       3929346
# 3       2432712
# 4       1178852
# 6       1174647
# 0        195416
# 208          64
# 9            23
# 7            15
# 8             9
# 129           2
# 51            1
# 49            1
# 34            1
# Name: passenger_count, dtype: int64


"""
has nan
fare_amount has negative values (refunds?)
outliers in coordinates
208 pasengers?
"""

test_df.info()

test_df.describe()
#        pickup_longitude  pickup_latitude  dropoff_longitude  dropoff_latitude  \
# count       9914.000000      9914.000000        9914.000000       9914.000000
# mean         -73.974722        40.751041         -73.973657         40.751743
# std            0.042774         0.033542           0.039072          0.035435
# min          -74.252190        40.573143         -74.263245         40.568974
# 25%          -73.992500        40.736125         -73.991250         40.735253
# 50%          -73.982325        40.753050         -73.980015         40.754064
# 75%          -73.968013        40.767113         -73.964061         40.768757
# max          -72.986534        41.709557         -72.990970         41.696682

#        passenger_count         year        month      weekday          day  \
# count      9914.000000  9914.000000  9914.000000  9914.000000  9914.000000
# mean          1.671273  2011.815312     6.861307     2.842243    16.141921
# std           1.278747     1.803435     3.353546     1.937365     8.821059
# min           1.000000  2009.000000     1.000000     0.000000     1.000000
# 25%           1.000000  2010.000000     4.000000     1.000000     9.000000
# 50%           1.000000  2012.000000     7.000000     3.000000    16.000000
# 75%           2.000000  2014.000000    10.000000     5.000000    25.000000
# max           6.000000  2015.000000    12.000000     6.000000    31.000000

#               hour
# count  9914.000000
# mean     12.404983
# std       6.031902
# min       0.000000
# 25%       8.000000
# 50%      13.000000
# 75%      17.000000
# max      23.000000

""" # NO NaN"""
test_df.isnull().sum()

test_df['passenger_count'].value_counts()
# 1    6914
# 2    1474
# 5     696
# 3     447
# 4     206
# 6     177
# Name: passenger_count, dtype: int64

# %% label = fare_amount

train_df = train_df_raw.head(100)

"""
The taxi fare estimate within NYC, from the Bronx to Staten Island is around $120
base fare is $2.50

"""

train_df['fare_amount'].describe()
# count    5.542386e+07
# mean     8.077921e+00
# std      2.055127e+01
# min     -3.000000e+02
# 25%      6.000000e+00
# 50%      8.500000e+00
# 75%      1.250000e+01
# max      9.396336e+04
# Name: fare_amount, dtype: float64

"""
outliers: negative values? max = 9.396336e+04?
"""
fig = plt.figure('fare_amount_training_all', dpi=150)
sns.boxplot(x=train_df['fare_amount'], color='skyblue')
ax = plt.gca()
ax.set_xlabel('Fare amount in USD')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xlim(-0.1,)
# plt.savefig('plots/training dataset/fare_amount_training_all.png', dpi = 150)

fig = plt.figure('fare_amount_training_all_hist', dpi=150)
train_df[['fare_amount']].hist(
    bins=100, grid=False, ax=plt.gca(), color='skyblue')
plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Fare amount in USD')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0.1,)
# plt.savefig('plots/training dataset/fare_amount_training_all_hist.png', dpi = 150)

# explore outliers

len(train_df['fare_amount'][train_df['fare_amount'] > 2000])
# 8
len(train_df['fare_amount'][train_df['fare_amount'] < 2.50])
# 4747

fig = plt.figure('fare_amount_training_2.5_2000', dpi=150)
sns.boxplot(x=train_df['fare_amount'][(train_df['fare_amount'] < 2000) & (
    train_df['fare_amount'] > 2.49)], color='skyblue')
ax = plt.gca()
ax.set_xlabel('Fare amount in USD')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xlim(-0.1,)
# plt.savefig('plots/training dataset/fare_amount_2.5_2000.png', dpi = 150)


fig = plt.figure('fare_amount_training_2.5_2000_hist', dpi=150)
train_df['fare_amount'][(train_df['fare_amount'] < 2000) & (
    train_df['fare_amount'] > 2.49)].hist(bins=100, grid=False, ax=plt.gca(), color='skyblue')
plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Fare amount in USD')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0.1,)
# plt.savefig('plots/training dataset/fare_amount_2.5_2000_hist.png', dpi = 150)

len(train_df['fare_amount'][train_df['fare_amount'] > 800])
# 21


"""
from now on, we discard the outliers from the training set; we set upper bound to be 800
"""
# !!!
train_df = train_df[(train_df['fare_amount'] < 800) &
                    (train_df['fare_amount'] > 2.49)]

print(len(train_df))
# 55423856


fig = plt.figure('fare_amount_training_2.5_800', dpi=150)
sns.boxplot(x=train_df['fare_amount'], color='skyblue')
ax = plt.gca()
ax.set_xlabel('Fare amount in USD')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xlim(-0.1,)
# plt.savefig('plots/training dataset/fare_amount_2.5_800.png', dpi = 150)


fig = plt.figure('fare_amount_training_2.5_800_hist', dpi=150)
train_df['fare_amount'].hist(
    bins=100, grid=False, ax=plt.gca(), color='skyblue')
plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Fare amount in USD')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0.1,)
# plt.savefig('plots/training dataset/fare_amount_2.5_800_hist.png', dpi = 150)


# %% passenger_count

train_df['passenger_count'].value_counts()
# 1      38334270
# 2       8174491
# 5       3929071
# 3       2432486
# 4       1178731
# 6       1174554
# 0        195369
# 208          64
# 9            23
# 7            15
# 8             9
# 129           2
# 51            1
# 49            1
# 34            1
# Name: passenger_count, dtype: int64

test_df['passenger_count'].value_counts()
# 1    6914
# 2    1474
# 5     696
# 3     447
# 4     206
# 6     177
# Name: passenger_count, dtype: int64

"""
test set has NO 0 passengers too; CAN discard
"""

# train_df = train_df[train_df['passenger_count'] > 0.9]

train_df_passenger_count00 = train_df[['passenger_count']].groupby(
    ['passenger_count']).size()

fig = plt.figure('passenger_count_hist', dpi=150)
sns.barplot(x=train_df_passenger_count00.index,
            y=train_df_passenger_count00.values, color='skyblue')
plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Number of passengers')
ax.set_ylabel(None)
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig('plots/training dataset/count by number of passengers.png', dpi = 150)

train_df_passenger_count0 = train_df[['passenger_count']][(train_df['passenger_count'] > 0) & (
    train_df['passenger_count'] < 10)].groupby(['passenger_count']).size()

fig = plt.figure('passenger_count_g0_leq9_hist', dpi=150)
sns.barplot(x=train_df_passenger_count0.index,
            y=train_df_passenger_count0.values, color='skyblue')
plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Number of passengers')
ax.set_ylabel(None)
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig('plots/training dataset/count by number of passengers_g0_leq9.png', dpi = 150)

train_df_passenger_count = train_df[['passenger_count']][(train_df['passenger_count'] > 0) & (
    train_df['passenger_count'] < 7)].groupby(['passenger_count']).size()

fig = plt.figure('passenger_count_g0_leq6_hist', dpi=150)
sns.barplot(x=train_df_passenger_count.index,
            y=train_df_passenger_count.values, color='skyblue')
plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Number of passengers')
ax.set_ylabel(None)
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig('plots/training dataset/count by number of passengers_g0_leq6.png', dpi = 150)

"""
mostly 1 passenger
"""

# test set
test_df_passenger_count00 = test_df[['passenger_count']].groupby(
    ['passenger_count']).size()

fig = plt.figure('passenger_count_hist', dpi=150)
sns.barplot(x=test_df_passenger_count00.index,
            y=test_df_passenger_count00.values, color='tomato')
plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Number of passengers')
ax.set_ylabel(None)
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig('plots/test dataset/count by number of passengers.png', dpi = 150)

"""
1-6 passengers in test set
"""

train_df_passenger_count = train_df[['passenger_count']][(train_df['passenger_count'] > 0) & (
    train_df['passenger_count'] < 7)].groupby(['passenger_count']).size()

figure, axes = plt.subplots(1, 2, dpi=150)
sns.barplot(x=train_df_passenger_count.index,
            y=train_df_passenger_count.values, color='skyblue', ax=axes[0])
sns.barplot(x=test_df_passenger_count00.index,
            y=test_df_passenger_count00.values, color='tomato', ax=axes[1])
axes[0].set_yscale('log')
axes[1].set_yscale('log')
axes[0].set_xlabel('Number of passengers')
axes[0].set_ylabel(None)
axes[0].set_title(None)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[1].set_xlabel('Number of passengers')
axes[1].set_ylabel(None)
axes[1].set_title(None)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
# plt.savefig('plots/test dataset/count by number of passengers_train_test.png', dpi = 150)


# !!!!
""" # only keep passenger 1-6 """
train_df = train_df[(train_df['passenger_count'] > 0) &
                    (train_df['passenger_count'] < 7)]


"""
fare by number of passengers
"""
train_df_passenger_fare = train_df[['passenger_count', 'fare_amount']].groupby(
    ['passenger_count']).agg(np.nanmean).reset_index()

fig = plt.figure('fare_by_passenger_bar', dpi=150)
sns.barplot(x=train_df_passenger_fare['passenger_count'],
            y=train_df_passenger_fare['fare_amount'], color='skyblue')
# plt.yscale('log')
ax = plt.gca()
# for i, p in enumerate(ax.patches):
#     height = p.get_height()
#     ax.text(p.get_x() + p.get_width()/2., height + 0.1, round(train_df_passenger_fare['fare_amount'][i], 2), ha = "center", size = 'small')
ax.set_xlabel('Number of passengers')
ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(10.5, 12)
# ax.spines['left'].set_visible(False)
# plt.savefig('plots/training dataset/fare by number of passengers.png', dpi = 150)


fig = plt.figure('fare_by_passenger_g0_leq6_bar', dpi=150)
sns.barplot(x=train_df_passenger_fare['passenger_count'][(train_df_passenger_fare['passenger_count'] > 0) & (
    train_df_passenger_fare['passenger_count'] < 7)], y=train_df_passenger_fare['fare_amount'][train_df_passenger_fare['passenger_count'] < 10], color='skyblue')
# plt.yscale('log')
ax = plt.gca()
# for i, p in enumerate(ax.patches):
#     height = p.get_height()
#     ax.text(p.get_x() + p.get_width()/2., height + 0.1, round(train_df_passenger_fare['fare_amount'][i], 2), ha = "center", size = 'small')
ax.set_xlabel('Number of passengers')
ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(10.5, 12)
# ax.spines['left'].set_visible(False)
# plt.savefig('plots/training dataset/fare by number of passengers_g0_leq6.png', dpi = 150)


"""
duration 
"""

"""
use  encoding or OH

"""

# %% pickup datetimes

"""
year
"""
# def get_year(row):
#     return row.year
# train_df['year'] = train_df['pickup_datetime'].apply(get_year)

train_df_year_count = train_df[['year']].groupby(['year']).size()

fig = plt.figure('year_count_training', dpi=150)
sns.barplot(x=train_df_year_count.index,
            y=train_df_year_count.values, color='skyblue')
ax = plt.gca()
ax.set_xlabel('Year')
# for i, p in enumerate(ax.patches):
#     height = p.get_height()
#     ax.text(p.get_x() + p.get_width()/2., height + 0.1, round(train_df_year_count[i], 2), ha = "center", size = 'small')
# ax.bar_label(ax.containers[0])
ax.set_ylabel(None)
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(0,)
# plt.savefig('plots/training dataset/year_count_training.png', dpi = 150)
"""
2008-2015
"""

# test set
test_df_year_count = test_df[['year']].groupby(['year']).size()

fig = plt.figure('year_count_training', dpi=150)
sns.barplot(x=test_df_year_count.index,
            y=test_df_year_count.values, color='tomato')
ax = plt.gca()
ax.set_xlabel('Year')
# for i, p in enumerate(ax.patches):
#     height = p.get_height()
#     ax.text(p.get_x() + p.get_width()/2., height + 0.1, round(train_df_year_count[i], 2), ha = "center", size = 'small')
# ax.bar_label(ax.containers[0])
ax.set_ylabel(None)
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(0,)
# plt.savefig('plots/test dataset/year_count_training.png', dpi = 150)
"""
2009-2015. Drop 2008 from training set!
"""

train_df_year_count1 = train_df_year_count.drop(2008, axis=0)

figure, axes = plt.subplots(1, 2, dpi=150)
sns.barplot(x=train_df_year_count1.index,
            y=train_df_year_count1.values, color='skyblue', ax=axes[0])
sns.barplot(x=test_df_year_count.index,
            y=test_df_year_count.values, color='tomato', ax=axes[1])
ax = axes[0]
ax.set_xlabel('Year')
ax.set_ylabel(None)
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(0,)
ax = axes[1]
ax.set_xlabel('Year')
ax.set_ylabel(None)
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(0,)
# plt.savefig('plots/test dataset/count by year_train_test.png', dpi = 150)


"""
mean trip_duration by year
"""
train_df_yearly_fare = train_df[['year', 'fare_amount']].groupby(
    ['year']).agg(np.nanmean).reset_index()

#    year  fare_amount
# 0  2008    10.633574
# 1  2009    10.055845
# 2  2010    10.163314
# 3  2011    10.445408
# 4  2012    11.176920
# 5  2013    12.597136
# 6  2014    12.920580
# 7  2015    12.987608


fig = plt.figure('year_mean_fare_training', dpi=150)
sns.barplot(x=train_df_yearly_fare['year'],
            y=train_df_yearly_fare['fare_amount'], color='skyblue')
ax = plt.gca()
ax.set_xlabel('Year')
# for i, p in enumerate(ax.patches):
#     height = p.get_height()
#     ax.text(p.get_x() + p.get_width()/2., height + 0.1, round(train_df_yearly_fare['fare_amount'][i], 2), ha = "center", size = 'small')
# ax.bar_label(ax.containers[0])
ax.set_ylabel(None)
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(9.5,)
# plt.savefig('plots/training dataset/year_mean_fare_training.png', dpi = 150)


"""
month
"""
# def get_month(row):
#     return row.month
# train_df['month'] = train_df['pickup_datetime'].apply(get_month)


train_df_month_count = train_df[['month']].groupby(['month']).size()

fig = plt.figure('month_count_training', dpi=150)
sns.barplot(x=train_df_month_count.index,
            y=train_df_month_count.values, color='skyblue')
ax = plt.gca()
ax.set_xlabel('Month')
# for i, p in enumerate(ax.patches):
#     height = p.get_height()
#     ax.text(p.get_x() + p.get_width()/2., height + 0.1, round(train_df_month_count.values[i], 2), ha = "center", size = 'small')
# ax.bar_label(ax.containers[0])
ax.set_ylabel(None)
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(3.5e6,)
# plt.savefig('plots/training dataset/month_count_training.png', dpi = 150)

"""
basically uniformly distributed among Jan-Jun
"""


"""
mean trip_duration by month
"""
train_df_monthly_fare = train_df[['month', 'fare_amount']].groupby(
    ['month']).agg(np.nanmean).reset_index()

fig = plt.figure('month_mean_fare_training', dpi=150)
sns.barplot(x=train_df_monthly_fare['month'],
            y=train_df_monthly_fare['fare_amount'], color='skyblue')
ax = plt.gca()
ax.set_xlabel('Month')
# for i, p in enumerate(ax.patches):
#     height = p.get_height()
#     ax.text(p.get_x() + p.get_width()/2., height + 0.1, round(train_df_monthly_fare['fare_amount'][i], 2), ha = "center", size = 'small')
# ax.bar_label(ax.containers[0])
ax.set_ylabel(None)
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(10, 12)
# plt.savefig('plots/training dataset/month_mean_fare_training.png', dpi = 150)


"""
day of the week; 0 is Monday
"""
# def get_weekday(row):
#     return row.weekday()
# train_df['weekday'] = train_df['pickup_datetime'].apply(get_weekday)

train_df_weekday_count = train_df[['weekday']].groupby(['weekday']).size()

fig = plt.figure('weekday_count_training', dpi=150)
sns.barplot(x=train_df_weekday_count.index,
            y=train_df_weekday_count.values, color='skyblue')
ax = plt.gca()
ax.set_xlabel('Weekday')
# for i, p in enumerate(ax.patches):
#     height = p.get_height()
#     ax.text(p.get_x() + p.get_width()/2., height + 0.1, round(train_df_weekday_count[i], 2), ha = "center", size = 'x-small')
# ax.bar_label(ax.containers[0])
ax.set_ylabel(None)
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(5e6,)
# plt.savefig('plots/training dataset/weekday_count_training.png', dpi = 150)


"""
mean trip_duration by weekday
"""
train_df_weekday_fare = train_df[['weekday', 'fare_amount']].groupby(
    ['weekday']).agg(np.nanmean).reset_index()
train_df_weekday_fare['weekday'] = train_df_weekday_fare['weekday'].replace(
    dict(zip(range(0, 7), ['Mon', 'Tue', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])))

fig = plt.figure('weekday_mean_fare_training', dpi=150)
sns.barplot(x=train_df_weekday_fare['weekday'],
            y=train_df_weekday_fare['fare_amount'], color='skyblue')
ax = plt.gca()
ax.set_xlabel('Weekday')
# for i, p in enumerate(ax.patches):
#     height = p.get_height()
#     ax.text(p.get_x() + p.get_width()/2., height + 0.1, round(train_df_weekday_fare['fare_amount'][i], 2), ha = "center", size = 'small')
# ax.bar_label(ax.containers[0])
ax.set_ylabel(None)
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(10.5, 12)
# plt.savefig('plots/training dataset/weekday_mean_fare_training.png', dpi = 150)


"""
day of the month
"""
# def get_day(row):
#     return row.day
# train_df['day'] = train_df['pickup_datetime'].apply(get_day)

train_df_day_count = train_df[['day']].groupby(['day']).size()

fig = plt.figure('day_count_training', dpi=150)
sns.barplot(x=train_df_day_count.index,
            y=train_df_day_count.values, color='skyblue')
ax = plt.gca()
ax.set_xlabel('Day')
# for i, p in enumerate(ax.patches):
#     height = p.get_height()
#     ax.text(p.get_x() + p.get_width()/2., height + 0.1, round(train_df_day_count[i], 2), ha = "center", size = 'x-small', rotation = 45)
# ax.bar_label(ax.containers[0])
ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_xticks([0, 14, 29])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(1e6,)
# plt.savefig('plots/training dataset/day_count_training.png', dpi = 150)

"""
mean trip_duration by day
"""

train_df_day_fare = train_df[['day', 'fare_amount']].groupby(
    ['day']).agg(np.nanmean).reset_index()

fig = plt.figure('day_mean_fare_training', dpi=150)
sns.barplot(x=train_df_day_fare['day'],
            y=train_df_day_fare['fare_amount'], color='skyblue')
ax = plt.gca()
ax.set_xlabel('Day')
# for i, p in enumerate(ax.patches):
#     height = p.get_height()
#     ax.text(p.get_x() + p.get_width()/2., height + 0.1, round(train_df_day_fare['fare_amount'][i], 2), ha = "center", size = 'small', rotation = 45)
# ax.bar_label(ax.containers[0])
ax.set_ylabel(None)
ax.set_xticks([0, 14, 29])
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(11.2, 11.45)
# plt.savefig('plots/training dataset/day_mean_fare_training.png', dpi = 150)


"""
hour of the day
"""
# def get_hour(row):
#     return row.hour
# train_df['hour'] = train_df['pickup_datetime'].apply(get_hour)

train_df_hour_count = train_df[['hour']].groupby(['hour']).size()

fig = plt.figure('day_count_training', dpi=150)
sns.barplot(x=train_df_hour_count.index,
            y=train_df_hour_count.values, color='skyblue')
ax = plt.gca()
ax.set_xlabel('Hour')
# for i, p in enumerate(ax.patches):
#     height = p.get_height()
#     ax.text(p.get_x() + p.get_width()/2., height + 0.1, round(train_df_hour_count[i], 2), ha = "center", size = 'x-small', rotation = 45)
# ax.bar_label(ax.containers[0])
ax.set_ylabel(None)
ax.set_xticks(np.arange(0, 25, 2))
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(0,)
# plt.savefig('plots/training dataset/hour_count_training.png', dpi = 150)


"""
mean trip_duration by hour
"""

train_df_hour_fare = train_df[['hour', 'fare_amount']].groupby(
    ['hour']).agg(np.nanmean).reset_index()

fig = plt.figure('hour_mean_fare_training', dpi=150)
sns.barplot(x=train_df_hour_fare['hour'],
            y=train_df_hour_fare['fare_amount'], color='skyblue')
ax = plt.gca()
ax.set_xlabel('Hour')
# for i, p in enumerate(ax.patches):
#     height = p.get_height()
#     ax.text(p.get_x() + p.get_width()/2., height + 0.1, round(train_df_day_fare['fare_amount'][i], 2), ha = "center", size = 'small', rotation = 45)
# ax.bar_label(ax.containers[0])
ax.set_ylabel(None)
ax.set_xticks(np.arange(0, 25, 2))
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(9.5, 14.5)
# plt.savefig('plots/training dataset/hour_mean_fare_training.png', dpi = 150)

# !!!
train_df = train_df[train_df['year'] != 2008]
len(train_df)
# 55220148


# %% coordinates

"""
all nan were gone with the outliers
"""

# pickup coordinates had no nan
train_df['pickup_latitude'].describe()
# count    5.522015e+07
# mean     1.944475e+01
# std      1.416285e+01
# min     -3.492264e+03
# 25%      4.073493e+01
# 50%      4.075265e+01
# 75%      4.076713e+01
# max      3.408790e+03
# Name: pickup_latitude, dtype: float64

fig = plt.figure('pickup_latitude_training', dpi=150)
sns.boxplot(x=train_df['pickup_latitude'], color='skyblue')
ax = plt.gca()
ax.set_xlabel('Pickup latitude')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xlim(40.40, 41.06)
# plt.savefig('plots/training dataset/geodata/pickup_latitude_training3.png', dpi = 150)


fig = plt.figure('pickup_latitude_training', dpi=150)
train_df[['pickup_latitude']].hist(
    bins=50, grid=False, ax=plt.gca(), color='skyblue', label='training set')
test_df[['pickup_latitude']].hist(
    bins=100, grid=False, ax=plt.gca(), color='tomato', label='test set')
plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Pickup latitude')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(40.4, 41.06)
ax.legend()
# plt.savefig('plots/training dataset/pickup_latitude_training.png', dpi = 150)

lat_min = 40.40
lat_max = 41.06

len(train_df['pickup_latitude'][((train_df['pickup_latitude'] < 40.40) | (
    train_df['pickup_latitude'] > 41.06)) & (train_df['pickup_latitude'] != 0.0)])
# 68038
len(train_df['pickup_latitude'][(train_df['pickup_latitude'] == 0.0)])
# 1048130
"""
many zeros
"""

fig = plt.figure('pickup_longitude', dpi=150)
sns.boxplot(x=train_df['pickup_longitude'], color='skyblue')
ax = plt.gca()
ax.set_xlabel('Pickup longitude')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xlim(-75.00, -73.33)
# plt.savefig('plots/training dataset/geodata/pickup_longitude_training2.png', dpi = 150)
long_max = -73.33
long_min = -75.00
len(train_df['pickup_longitude'][((train_df['pickup_longitude'] > -73.33) |
    (train_df['pickup_longitude'] < -75.00)) & (train_df['pickup_longitude'] != 0.0)])
# 64197
len(train_df['pickup_longitude'][(train_df['pickup_longitude'] == 0.0)])
# 1051666

fig = plt.figure('pickup_longitude_training', dpi=150)
train_df[['pickup_longitude']].hist(
    bins=100, grid=False, ax=plt.gca(), color='skyblue', label='training set')
test_df[['pickup_longitude']].hist(
    bins=100, grid=False, ax=plt.gca(), color='tomato', label='test set')
plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Pickup longitude')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-75., -73.33)
# plt.savefig('plots/training dataset/pickup_longitude_training.png', dpi = 150)

"""
many zeros
"""

fig = plt.figure('dropoff_latitude_training', dpi=150)
sns.boxplot(x=train_df['dropoff_latitude'], color='skyblue')
ax = plt.gca()
ax.set_xlabel('Dropoff latitude')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xlim(40.40, 41.06)
# plt.savefig('plots/training dataset/geodata/dropoff_latitude_training3.png', dpi = 150)

fig = plt.figure('dropoff_latitude_training', dpi=150)
train_df[['dropoff_latitude']].hist(
    bins=100, grid=False, ax=plt.gca(), color='skyblue', label='training set')
test_df[['dropoff_latitude']].hist(
    bins=100, grid=False, ax=plt.gca(), color='tomato', label='test set')
plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Dropoff latitude')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(40.40, 41.06)
ax.legend()
# plt.savefig('plots/training dataset/dropoff_latitude_training.png', dpi = 150)

lat_min = 40.40
lat_max = 41.06

len(train_df['dropoff_latitude'][((train_df['dropoff_latitude'] < 40.40) | (
    train_df['dropoff_latitude'] > 41.06)) & (train_df['dropoff_latitude'] != 0.0)])
# 66259
len(train_df['dropoff_latitude'][(train_df['dropoff_latitude'] == 0.0)])
# 1045404

# train_df0 = train_df.head(1000)

train_dfm = train_df[['pickup_latitude', 'dropoff_latitude']].melt(
    var_name='dataset').replace({'pickup_latitude': 'Pickup', 'dropoff_latitude': 'Dropoff'})

fig = plt.figure(dpi=150)
sns.boxplot(x="value", data=train_dfm, y='dataset')
ax = plt.gca()
ax.set_xlabel('Latitude')
ax.set_ylabel(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xlim(40.65, 40.85)
# plt.savefig('plots/training dataset/geodata/pickup_dropoff_latitude_training.png', dpi = 150)


fig = plt.figure('dropoff_longitude', dpi=150)
sns.boxplot(x=train_df['dropoff_longitude'], color='skyblue')
ax = plt.gca()
ax.set_xlabel('Dropoff longitude')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xlim(-75.00, -73.33)
# plt.savefig('plots/training dataset/geodata/dropoff_longitude_training2.png', dpi = 150)
long_max = -73.33
long_min = -75.00
len(train_df['dropoff_longitude'][((train_df['dropoff_longitude'] > -73.33) |
    (train_df['dropoff_longitude'] < -75.00)) & (train_df['dropoff_longitude'] != 0.0)])
# 58303
len(train_df['dropoff_longitude'][(train_df['dropoff_longitude'] == 0.0)])
# 1048483

fig = plt.figure('dropoff_longitude', dpi=150)
train_df[['dropoff_longitude']].hist(
    bins=100, grid=False, ax=plt.gca(), color='skyblue', label='training set')
test_df[['dropoff_longitude']].hist(
    bins=100, grid=False, ax=plt.gca(), color='tomato', label='test set')
plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Dropoff longitude')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
# plt.xlim(-0.1,)
# plt.savefig('plots/training dataset/dropoff_longitude_training.png', dpi = 150)

train_dfm = train_df[['pickup_longitude', 'dropoff_longitude']].melt(
    var_name='dataset').replace({'pickup_longitude': 'Pickup', 'dropoff_longitude': 'Dropoff'})

fig = plt.figure(dpi=150)
sns.boxplot(x="value", data=train_dfm, y='dataset')
ax = plt.gca()
ax.set_xlabel('Longitude')
ax.set_ylabel(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xlim(-74.2, -73.8)
# plt.savefig('plots/training dataset/geodata/pickup_dropoff_longitude_training.png', dpi = 150)


# test set

fig = plt.figure(dpi=150)
sns.boxplot(x=test_df['pickup_latitude'], color='skyblue')
ax = plt.gca()
ax.set_xlabel('Pickup latitude')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xlim(40.40, 41.06)
# plt.savefig('plots/test dataset/pickup_latitude_test.png', dpi = 150)

lat_min = 40.40
lat_max = 41.06

len(test_df['pickup_latitude'][((test_df['pickup_latitude'] < 40.40) | (
    test_df['pickup_latitude'] > 41.06)) & (test_df['pickup_latitude'] != 0.0)])
# 9
len(test_df['pickup_latitude'][(test_df['pickup_latitude'] == 0.0)])
# 0
"""
many zeros
"""

fig = plt.figure('pickup_longitude', dpi=150)
sns.boxplot(x=test_df['pickup_longitude'], color='skyblue')
ax = plt.gca()
ax.set_xlabel('Pickup longitude')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xlim(-75.00, -73.33)
# plt.savefig('plots/test dataset/pickup_longitude_test.png', dpi = 150)
long_max = -73.33
long_min = -75.00
len(test_df['pickup_longitude'][((test_df['pickup_longitude'] > -73.33) |
    (test_df['pickup_longitude'] < -75.00)) & (test_df['pickup_longitude'] != 0.0)])
# 8
len(test_df['pickup_longitude'][(test_df['pickup_longitude'] == 0.0)])
# 0

fig = plt.figure('dropoff_latitude', dpi=150)
sns.boxplot(x=test_df['dropoff_latitude'], color='skyblue')
ax = plt.gca()
ax.set_xlabel('Dropoff latitude')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xlim(40.40, 41.06)
# plt.savefig('plots/test dataset/dropoff_latitude_test.png', dpi = 150)


fig = plt.figure('dropoff_longitude', dpi=150)
sns.boxplot(x=test_df['dropoff_longitude'], color='skyblue')
ax = plt.gca()
ax.set_xlabel('Dropoff longitude')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xlim(-75.00, -73.33)
# plt.savefig('plots/test dataset/dropoff_longitude_test.png', dpi = 150)


# %% displacement

# train_df = train_df_raw.copy()

# !!!
# approximate coordinate boundaries of NYC; includes EWR and some parts of NJ, LI
lat_min = 40.40
lat_max = 41.06
long_max = -73.33
long_min = -75.00

# only consider locations within NYC
train_df = train_df[
    (train_df['pickup_latitude'] > lat_min)
    & (train_df['pickup_latitude'] < lat_max)
    & (train_df['pickup_longitude'] > long_min)
    & (train_df['pickup_longitude'] < long_max)
    & (train_df['dropoff_latitude'] > lat_min)
    & (train_df['dropoff_latitude'] < lat_max)
    & (train_df['dropoff_longitude'] > long_min)
    & (train_df['dropoff_longitude'] < long_max)]


# NYC map from NYC open data
nyc_zip_map = gpd.read_file(
    'nyc_opendata/ZIP_CODE_040114.shp').to_crs({'init': 'epsg:4326'})

# plot coordinates on the map
pickup_geometry = [Point(xy) for xy in zip(
    train_df['pickup_longitude'], train_df['pickup_latitude'])]
dropoff_geometry = [Point(xy) for xy in zip(
    train_df['dropoff_longitude'], train_df['dropoff_latitude'])]

pickup_geo_df = gpd.GeoDataFrame(geometry=pickup_geometry)
dropoff_geo_df = gpd.GeoDataFrame(geometry=dropoff_geometry)

pickup_geo_df.to_csv('engineered_datasets/pickup_geo_df.csv')
dropoff_geo_df.to_csv('engineered_datasets/dropoff_geo_df.csv')


fig, ax = plt.subplots(figsize=(15, 15))
nyc_zip_map.plot(ax=ax, color='grey', alpha=1.0, zorder=1)
ax = plt.gca()
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
pickup_geo_df.plot(ax=ax, markersize=20, color='red',
                   marker='o', zorder=2, label='pickup location')
dropoff_geo_df.plot(ax=ax, markersize=20, color='blue',
                    marker='^', zorder=3, label='dropoff location')
plt.legend()
# plt.savefig('plots/NYCmap', dpi = 150)


"""
haversine distance (too slow!); use Euclidean distance (they are almost the same)
"""
# from haversine import haversine, Unit

# def haversine_distance_in_miles(row):
#     lat1 = row['pickup_latitude']
#     long1 = row['pickup_longitude']
#     lat2 = row['dropoff_latitude']
#     long2 = row['dropoff_longitude']
#     row['haversine_distance_miles'] = haversine((lat1, long1), (lat2, long2), unit = Unit.MILES)
#     return row


R_earth = 3959.  # radius of the earth in miles = 3959. feature will be rescaled, so set it to 1
train_df['long_displacement'] = R_earth * ((train_df['dropoff_longitude'] - train_df['pickup_longitude'])
                                           * np.pi / 180) * np.cos(0.5*(train_df['dropoff_latitude'] + train_df['pickup_latitude']) * np.pi / 180)
train_df['lat_displacement'] = abs(
    R_earth * (train_df['dropoff_latitude'] - train_df['pickup_latitude']) * np.pi / 180)
train_df['euclidean_distance_miles'] = np.sqrt(
    train_df['long_displacement'] * train_df['long_displacement'] + train_df['lat_displacement'] * train_df['lat_displacement'])

test_df['long_displacement'] = R_earth * ((test_df['dropoff_longitude'] - test_df['pickup_longitude'])
                                          * np.pi / 180) * np.cos(0.5*(test_df['dropoff_latitude'] + test_df['pickup_latitude']) * np.pi / 180)
test_df['lat_displacement'] = abs(
    R_earth * (test_df['dropoff_latitude'] - test_df['pickup_latitude']) * np.pi / 180)
test_df['euclidean_distance_miles'] = np.sqrt(
    test_df['long_displacement'] * test_df['long_displacement'] + test_df['lat_displacement'] * test_df['lat_displacement'])

train_df['euclidean_distance_miles'].head()

# 0    0.931195
# 1    1.121959
# 2    3.967761
# 3    0.923103
# 4    0.738600
# Name: euclidean_distance_miles, dtype: float64

train_df['euclidean_distance_miles'].describe()
# count    1.447129e+06
# mean     2.139597e+00
# std      2.415418e+00
# min      0.000000e+00
# 25%      7.733966e-01
# 50%      1.308670e+00
# 75%      2.417073e+00
# max      2.803162e+01
# Name: euclidean_distance_miles, dtype: float64


fig = plt.figure('euclidean_distance_miles_hist', dpi=150)
train_df['euclidean_distance_miles'].hist(bins=100, grid=False, range=(
    0, 100), ax=plt.gca(), color='skyblue', label='training set')
test_df['euclidean_distance_miles'].hist(bins=100, grid=False, range=(
    0, 100), ax=plt.gca(), color='tomato', label='test set')
plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Euclidean distance in miles')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
# plt.xlim(-0.1, 100)
# plt.savefig('plots/training dataset/euclidean_distance_miles_hist.png', dpi = 150)


"""
direction
"""

train_df['direction'] = np.arctan2(
    train_df['lat_displacement'], train_df['long_displacement'])

train_df['sin_direction'] = np.sin(train_df['direction'])
train_df['cos_direction'] = np.cos(train_df['direction'])


fig = plt.figure(dpi=150)
sns.scatterplot(x=train_df['sin_direction'],
                y=train_df['fare_amount'], color='skyblue')
ax = plt.gca()
ax.set_xlabel('Sine of bearing')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xlim(-75.00, -73.33)


sns.scatterplot(x=train_df['sin_direction'], y=train_df['trip_duration'])
plt.savefig('plots/training dataset/sin_direction_duration', dpi=150)
sns.scatterplot(x=train_df['cos_direction'], y=train_df['trip_duration'])
plt.savefig('plots/training dataset/cos_direction_duration', dpi=150)

"""
use sine and cosine because they are periodic
"""


# %% zip codes and boroughs

"""
zip codes and boroughs
"""

# Reverse Geocoding--- mapping (latitude, longitude) to address--- in a quick way, using a KDTree to search a known dataset of coordinates and zipcodes for the closest match
# database: US Census Gazatte at https://www.codementor.io/@bobhaffner/reverse-geocoding-bljjp5byw

gaz_zip = pd.read_csv('us_census_gazateer/2021_Gaz_zcta_national.txt',
                      delimiter='\t', dtype={'GEOID': 'str'})
gaz_zip.columns = gaz_zip.columns.str.strip()
gaz_zip = gaz_zip[['GEOID', 'INTPTLAT', 'INTPTLONG']]

# NYC opendata
dbf = Dbf5('nyc_opendata/ZIP_CODE_040114.dbf')
nyc_zip_county = dbf.to_dataframe()
nyc_zip_county.columns = nyc_zip_county.columns.str.strip()
nyc_zip_county = nyc_zip_county[['ZIPCODE', 'COUNTY']]

gaz_zip_county = pd.merge(left=gaz_zip, right=nyc_zip_county,
                          left_on='GEOID', right_on='ZIPCODE', how='left')

"""
use KDTree to map coordinates to nearest zipcode, then infer county
"""

kdt = KDTree(gaz_zip[['INTPTLAT', 'INTPTLONG']])


train_df['pickup_zipcode'] = gaz_zip_county.loc[kdt.query(
    train_df[['pickup_latitude', 'pickup_longitude']].to_numpy(), k=1, return_distance=False).squeeze(), 'GEOID'].values

train_df['pickup_county'] = gaz_zip_county.loc[kdt.query(
    train_df[['pickup_latitude', 'pickup_longitude']].to_numpy(), k=1, return_distance=False).squeeze(), 'COUNTY'].values

train_df['dropoff_zipcode'] = gaz_zip_county.loc[kdt.query(
    train_df[['dropoff_latitude', 'dropoff_longitude']].to_numpy(), k=1, return_distance=False).squeeze(), 'GEOID'].values

train_df['dropoff_county'] = gaz_zip_county.loc[kdt.query(
    train_df[['dropoff_latitude', 'dropoff_longitude']].to_numpy(), k=1, return_distance=False).squeeze(), 'COUNTY'].values

train_df[['pickup_county', 'dropoff_county']] = train_df[[
    'pickup_county', 'dropoff_county']].fillna('not_in_NYC')


train_df_pickup_county = train_df[['pickup_county']].groupby(
    ['pickup_county']).size()

fig = plt.figure('pickup_county', dpi=150)
sns.barplot(x=train_df_pickup_county.index,
            y=train_df_pickup_county.values, color='skyblue')
plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Pickup county')
ax.set_ylabel(None)
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig('plots/training dataset/geodata/pickup_county_count_training', dpi = 150)


train_df_dropoff_county = train_df[['dropoff_county']].groupby(
    ['dropoff_county']).size()

fig = plt.figure('pickup_county', dpi=150)
sns.barplot(x=train_df_dropoff_county.index,
            y=train_df_dropoff_county.values, color='skyblue')
plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Dropoff county')
ax.set_ylabel(None)
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig('plots/training dataset/geodata/dropoff_county_count', dpi = 150)
# mostly in Manhattan


"""
dropoff county == pickup county??
"""
train_df['dropoff_pickup_same_county'] = (
    train_df['dropoff_county'] == train_df['pickup_county'])


fig = plt.figure('same_county', dpi=150)
sns.countplot(train_df['dropoff_pickup_same_county'],
              orient='h', color='skyblue')
ax = plt.gca()
ax.set_xlabel('Dropoff county = Pickup county')
ax.set_ylabel(None)
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig('plots/training dataset/geodata/dropoff_pickup_same_county_count', dpi = 150)


"""
pickup_county, dropoff_county:
    OH encoding
"""


"""
zipcode: JFK LGA EWR
"""

# the above zipcode mapping was not exact; also include nearby zipcodes
airport_zipcode = ['11371', '11430', '11414', '11420', '11436', '11434', '11413', '11422', '11581', '11096',
                   '11369', '11370', '11372', '11105', '07114', '07201', '07105', '07208', '07112', '07108', '07102']

train_df['pickup_airport'] = train_df['pickup_zipcode'].isin(airport_zipcode)

# 7802
train_df['dropoff_airport'] = train_df['dropoff_zipcode'].isin(airport_zipcode)
# 18490


# uszeipcode and geopy are very slow on large datasets

# from uszipcode import SearchEngine
# search = SearchEngine(simple_zipcode=True)
# from uszipcode import Zipcode


# def convert_to_zip(lat, long):
#     result = search.by_coordinates(lat, long, radius=5, returns = 1)[0]
#     zipcode = result.zipcode
#     return zipcode

# train_df["pickup_zipcode"] = np.vectorize(convert_to_zip)(train_df["pickup_latitude"].values, train_df["pickup_longitude"].values)


# def get_zip_county(row):
#     pickup_lat = row['pickup_latitude']
#     pickup_long = row['pickup_longitude']
#     pickup_result = search.by_coordinates(pickup_lat, pickup_long, radius=5, returns = 1)[0]
#     row['pickup_zipcode'] = pickup_result.zipcode
#     row['pickup_county'] = pickup_result.county

#     dropoff_lat = row['dropoff_latitude']
#     dropoff_long = row['dropoff_longitude']
#     dropoff_result = search.by_coordinates(dropoff_lat, dropoff_long, radius=5, returns = 1)[0]
#     row['dropoff_zipcode'] = dropoff_result.zipcode
#     row['dropoff_county'] = dropoff_result.county

#     return row

# train_df = train_df.apply(get_zip_county, axis = 1)

sns.countplot(train_df['pickup_county'], orient='h', color='blue')
sns.countplot(train_df['dropoff_county'], orient='h', color='blue')

sns.countplot(train_df['dropoff_pickup_same_county'], orient='h', color='blue')

"""
duration by county
"""

# train_df_pickup_county_duration = train_df[['pickup_county', 'trip_duration']].groupby(['pickup_county']).agg(np.nanmean).reset_index()
# sns.barplot(x = train_df_pickup_county_duration['pickup_county'], y = train_df_pickup_county_duration['trip_duration'])
# # plt.savefig('plots/training dataset/train_df_passenger_duration.png', dpi = 150)

# train_df_dropoff_county_duration = train_df[['dropoff_county', 'trip_duration']].groupby(['dropoff_county']).agg(np.nanmean).reset_index()
# sns.barplot(x = train_df_dropoff_county_duration['dropoff_county'], y = train_df_dropoff_county_duration['trip_duration'])
# # plt.savefig('plots/training dataset/train_df_passenger_duration.png', dpi = 150)

# train_df_same_county_duration = train_df[['dropoff_pickup_same_county', 'trip_duration']].groupby(['dropoff_pickup_same_county']).agg(np.nanmean).reset_index()
# sns.barplot(x = train_df_same_county_duration['dropoff_pickup_same_county'], y = train_df_same_county_duration['trip_duration'])
# # plt.savefig('plots/training dataset/train_df_passenger_duration.png', dpi = 150)

# train_df_passenger_duration['trip_duration_minus_mean'] = train_df_passenger_duration['trip_duration'] - train_duration_mean


# sns.barplot(x = train_df_passenger_duration['passenger_count'], y = train_df_passenger_duration['trip_duration_minus_mean'])
# # plt.savefig('plots/training dataset/train_df_passenger_duration_by_passenger.png', dpi = 150)

# %% after dropping outliers
# !!!
"""
recall that we have dropped outliers by:
    train_df = train_df[(train_df['fare_amount'] < 800) & (train_df['fare_amount'] > 2.49)]
    train_df = train_df[(train_df['passenger_count'] > 0) & (train_df['passenger_count'] < 7)]
    train_df = train_df[train_df['year'] != 2008]
    
    train_df = train_df[
    (train_df['pickup_latitude'] > lat_min) 
    & (train_df['pickup_latitude'] < lat_max) 
    & (train_df['pickup_longitude'] > long_min) 
    & (train_df['pickup_longitude'] < long_max) 
    & (train_df['dropoff_latitude'] > lat_min) 
    & (train_df['dropoff_latitude'] < lat_max) 
    & (train_df['dropoff_longitude'] > long_min) 
    & (train_df['dropoff_longitude'] < long_max)]
"""

"""
The plots about fare amount, passenger count and datetime features remain very much the same.
Same for county plots

coordinate plots 
euclidean distance plot is smoothened

histograms of training and test sets follow very similar patterns. 

"""


# %% correlations

"""# correlation matrix"""


#         fare_amount  pickup_longitude  pickup_latitude  dropoff_longitude  \
# count  5.402855e+07      5.402855e+07     5.402855e+07       5.402855e+07
# mean   8.117405e+00     -3.974720e+01     1.987360e+01      -3.974720e+01
# std    9.694800e+00      2.521815e+01     1.260908e+01       2.521815e+01
# min    2.500000e+00     -7.498993e+01     4.040000e+01      -7.499828e+01
# 25%    6.000000e+00     -7.399229e+01     4.073656e+01      -7.399159e+01
# 50%    8.500000e+00     -7.398210e+01     4.075336e+01      -7.398061e+01
# 75%    1.250000e+01     -7.396832e+01     4.076753e+01      -7.396538e+01
# max    5.841600e+02     -7.333015e+01     4.105996e+01      -7.333009e+01

#        dropoff_latitude  passenger_count          year         month  \
# count      5.402855e+07     5.402855e+07  5.402855e+07  5.402855e+07
# mean       1.987360e+01     1.691272e+00  2.011738e+03  6.273112e+00
# std        1.260908e+01     1.307148e+00  1.865110e+00  3.436858e+00
# min        4.040000e+01     1.000000e+00  2.009000e+03  1.000000e+00
# 25%        4.073557e+01     1.000000e+00  2.010000e+03  3.000000e+00
# 50%        4.075385e+01     1.000000e+00  2.012000e+03  6.000000e+00
# 75%        4.076838e+01     2.000000e+00  2.013000e+03  9.000000e+00
# max        4.105997e+01     6.000000e+00  2.015000e+03  1.200000e+01

#             weekday           day          hour
# count  5.402855e+07  5.402855e+07  5.402855e+07
# mean   2.985013e+00  1.572349e+01  1.182271e+01
# std    1.905367e+00  8.683951e+00  5.835833e+00
# min    0.000000e+00  1.000000e+00  0.000000e+00
# 25%    1.000000e+00  8.000000e+00  7.000000e+00
# 50%    3.000000e+00  1.600000e+01  1.200000e+01
# 75%    5.000000e+00  2.300000e+01  1.700000e+01
# max    6.000000e+00  3.100000e+01  2.300000e+01


cor = train_df[['pickup_longitude', 'pickup_latitude',
                'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'year',
                'month', 'weekday', 'day', 'hour', 'daylight_saving',
                'long_displacement', 'lat_displacement', 'euclidean_distance_miles',
                'direction', 'sin_direction', 'cos_direction', 'pickup_zipcode',
                'pickup_county', 'dropoff_zipcode', 'dropoff_county',
                'dropoff_pickup_same_county', 'pickup_airport', 'dropoff_airport', 'fare_amount']].corr()

cor2 = train_df[['fare_amount', 'euclidean_distance_miles', 'pickup_longitude', 'pickup_latitude',
                 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'year',
                 'month', 'weekday', 'day', 'hour']].corr()
mask = np.array(cor2)
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots(dpi=200)
fig.set_size_inches(20, 30)
sns.heatmap(cor2, mask=mask, square=True, annot=True,
            annot_kws={'size': 'xx-small'})
ax = plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), fontsize='xx-small', rotation=40)
# plt.savefig('plots/training dataset/correlation_matrix.png', dpi = 200)

"""
pickup/drop off coordinates have not-so-small correlations with other things, including target label. Keep them
"""


# %% predictions


y_train = pd.read_csv('engineered_datasets/y_train.csv')
y_valid = pd.read_csv('engineered_datasets/y_valid.csv')
y_train_valid = pd.concat([y_train, y_valid], axis=0)
y_test_predict = pd.read_csv(
    'predictions/2021-09-09_3/y_test_predict_LGBMreg.csv')


fig = plt.figure('fare_amount_training_hist', dpi=200)
y_train_valid[y_train_valid['fare_amount'] < 80]['fare_amount'].hist(
    bins=100, grid=False, ax=plt.gca(), color='skyblue', label='training set')
y_test_predict[y_test_predict['fare_amount'] < 80]['fare_amount'].hist(
    bins=100, grid=False, ax=plt.gca(), color='tomato', label='test set (prediction)')
plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Fare amount in USD')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
plt.xlim(2.5, 80)
# plt.savefig('plots/fare_amount_trains_vs_test_predict_hist.png', dpi = 200)
