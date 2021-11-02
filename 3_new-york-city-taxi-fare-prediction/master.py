# -*- coding: utf-8 -*-
"""
Created on Wed Sep 08 19:00:00 2021

@author: Cedric Yu
"""

#%%
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

#%% Workflow

"""
Workflow

# load partly pre-processed training set containing month, weekday, day, hour extracted from parsed datetime

# diacard outliers before train-validation split: fare amount, passenger count, too far-off coordinates

# separate features and labels

# train-validation split

# pre-rocessing:

# euclidean distances and bearing

# extract zip codes and boroughs: Reverse Geocoding using KDTree with US Census Gazatte and NYC opendata sources

# identify which coordinates are in the 3 airports. fizzy match of zipcodes; we used KDTree to get the zipcode, which was not exact

# one-hot encoding: onehot_cols1 = ['passenger_count', 'pickup_county', 'dropoff_county']

# frequency encoding: freq_cols2 = ['weekday']

# mean encoding with y_train labels: target_mean_cols3 = ['year', 'month', 'day', 'hour', 'pickup_zipcode', 'dropoff_zipcode']

# cols_to_keep4 = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
       'dropoff_latitude', 'daylight_saving', 'long_displacement',
       'lat_displacement', 'euclidean_distance_miles', 
       'sin_direction', 'cos_direction', 'dropoff_pickup_same_county',
       'pickup_airport', 'dropoff_airport', 'passenger_count_1',
       'passenger_count_2', 'passenger_count_3', 'passenger_count_4',
       'passenger_count_5', 'passenger_count_6', 'pickup_county_Bronx',
       'pickup_county_Kings', 'pickup_county_New York', 'pickup_county_Queens',
       'pickup_county_Richmond', 'pickup_county_not_in_NYC',
       'dropoff_county_Bronx', 'dropoff_county_Kings',
       'dropoff_county_New York', 'dropoff_county_Queens',
       'dropoff_county_Richmond', 'dropoff_county_not_in_NYC',
       'weekday_freq_encoded', 'year_mean_encoded', 'month_mean_encoded',
       'day_mean_encoded', 'hour_mean_encoded', 'pickup_zipcode_mean_encoded',
       'dropoff_zipcode_mean_encoded']


# min-max rescaling # not used; we use tree-based models

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
import seaborn as sns
import gc

# import dask
# import dask.dataframe as dd


import os
os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\7_new-york-city-taxi-fare-prediction')


#%% load dataset train.csv
# !!!
"""
use partly pre-processed training set containing year, month, weekday, day, hour extracted from parsed datetime
"""

# get training set column names
train_cols_new = pd.read_csv('engineered_datasets/train_df_datetime.csv', nrows=0).columns
train_cols_new = train_cols_new.drop('Unnamed: 0').to_list()

# downcast datatypes to save RAM
dtypes_new = dict(zip(train_cols_new, [str, 'float32', 'float32', 'float32', 'float32', 'float32', 'uint8', 'uint16', 'uint8', 'uint8', 'uint8', 'uint8', bool]))

# import datetime-processed training dataset
train_df_raw = pd.read_csv('engineered_datasets/train_df_datetime.csv', low_memory = True, usecols = train_cols_new, dtype = dtypes_new)

train_df = train_df_raw.copy()

#%% diacard outliers before train-validation split

# we set bound of fare amount to be between 2.5 and 800
train_df = train_df[(train_df['fare_amount'] < 800) & (train_df['fare_amount'] > 2.49)]
# only keep passenger 1-6
train_df = train_df[(train_df['passenger_count'] > 0) & (train_df['passenger_count'] < 7)]
# drop data from year 2008
train_df = train_df[train_df['year'] != 2008]

# approximate coordinate boundaries of NYC, including EWR and parts of NJ and LI
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


#%% separate features and labels

train_df.reset_index(inplace = True)
X_train_valid = train_df.drop(['fare_amount'], axis = 1)
X_train_valid.drop(['index', 'key'], axis = 1, inplace = True)
y_train_valid = train_df['fare_amount']
"""# before you proceed: make sure the X and y have the same indices!!"""

del train_df, train_df_raw

#%% train-validation split

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, random_state = 1, train_size = 0.995)

del X_train_valid, y_train_valid


#%% pre-processing

#%% get month, weekday and day

"""datetime """

# def get_month(row):
#     return row.month

# def get_weekday(row):
#     return row.weekday()

# def get_day(row):
#     return row.day

# def get_hour(row):
#     return row.hour

# # to turn them into objects and then encode them below
# X_train['month'] = X_train['pickup_datetime'].apply(get_month).astype(object)
# X_train['weekday'] = X_train['pickup_datetime'].apply(get_weekday).astype(object)
# X_train['day'] = X_train['pickup_datetime'].apply(get_day).astype(object)
# X_train['hour'] = X_train['pickup_datetime'].apply(get_hour).astype(object)

# X_valid['month'] = X_valid['pickup_datetime'].apply(get_month).astype(object)
# X_valid['weekday'] = X_valid['pickup_datetime'].apply(get_weekday).astype(object)
# X_valid['day'] = X_valid['pickup_datetime'].apply(get_day).astype(object)
# X_valid['hour'] = X_valid['pickup_datetime'].apply(get_hour).astype(object)


#%% euclidean distances and bearing

"""# returns long_displacement, lat_displacement and euclidean_distance_miles"""

def euclidean_distance_miles(X):
    
    X_ = X.copy()
    R_earth = 1.  # radius of the earth in miles = 3959. feature will be rescaled, so set it to 1
    
    X_['long_displacement'] = (R_earth * ((X_['dropoff_longitude'] - X_['pickup_longitude']) * np.pi / 180) * np.cos( 0.5*(X_['dropoff_latitude'] + X_['pickup_latitude']) * np.pi / 180 )).astype('float32')
    X_['lat_displacement'] = (R_earth * (X_['dropoff_latitude'] - X_['pickup_latitude']) * np.pi / 180).astype('float32')
    X_['euclidean_distance_miles'] = (np.sqrt( X_['long_displacement'] * X_['long_displacement'] + X_['lat_displacement'] * X_['lat_displacement'] )).astype('float32')
    
    return X_


X_train = euclidean_distance_miles(X_train)
X_valid = euclidean_distance_miles(X_valid)
# verified with train_df

"""# direction of travel"""

X_train['direction'] = np.arctan2(X_train['lat_displacement'], X_train['long_displacement']).astype('float32')
X_train['sin_direction'] = np.sin(X_train['direction']).astype('float32')
X_train['cos_direction'] = np.cos(X_train['direction']).astype('float32')
X_train.drop(['direction'], axis = 1, inplace = True)

X_valid['direction'] = np.arctan2(X_valid['lat_displacement'], X_valid['long_displacement']).astype('float32')
X_valid['sin_direction'] = np.sin(X_valid['direction']).astype('float32')
X_valid['cos_direction'] = np.cos(X_valid['direction']).astype('float32')
X_valid.drop(['direction'], axis = 1, inplace = True)

#%% driving distance in meters

"""driving distance in meters"""

# not used; took forever
# """
# use osmnx and networkx to find driving distance
# we already found nearst nodes for all training and test set coordinates and saved the datasets for future use
# """

# import networkx as nx
# import osmnx as ox

# """# the map G uses the above coordinate boundaries of NYC; includes EWR and some parts of NJ, LI"""
# # G = ox.graph_from_bbox(lat_max, lat_min, long_max, long_min, network_type='drive')
# G = ox.io.load_graphml("nyc_driving_network.graphml")
# """# use strongly connected graph; otherwise some nodes are not conencted"""
# Gs = ox.utils_graph.get_largest_component(G, strongly=True)

# def driving_distance(row):
#     try:
#         row['driving_distance'] = nx.shortest_path_length(G, row['orig'], row['dest'], weight='length')
#     except nx.exception.NetworkXNoPath: # in case path is not found
#         row['driving_distance'] = np.nan
#         pass
        
#     return row

# X_train = X_train.apply(driving_distance, axis = 1)
# X_valid = X_valid.apply(driving_distance, axis = 1)

# X_train['driving_distance'].isnull().sum()
# X_valid['driving_distance'].isnull().sum()

# # fill nan with euclidean distance * mean of driving distance / mean of euclidean distance

# X_train['driving_distance'].fillna(X_train['euclidean_distance_miles'] * np.nanmean(X_train['driving_distance'])/ X_train['euclidean_distance_miles'].mean(), inplace = True)



#%% zip codes and boroughs

"""
zip codes and boroughs
"""

"""# Reverse Geocoding--- mapping (latitude, longitude) to address--- in a quick way, using a KDTree to search a known dataset of coordinates and zipcodes for the closest match"""

# database: US Census Gazatte at https://www.codementor.io/@bobhaffner/reverse-geocoding-bljjp5byw

gaz_zip = pd.read_csv('us_census_gazateer/2021_Gaz_zcta_national.txt', delimiter = '\t', dtype = {'GEOID' : 'str'})
gaz_zip.columns = gaz_zip.columns.str.strip()
gaz_zip = gaz_zip[['GEOID', 'INTPTLAT', 'INTPTLONG']]

from simpledbf import Dbf5
dbf = Dbf5('nyc_opendata/ZIP_CODE_040114.dbf')
nyc_zip_county = dbf.to_dataframe()
nyc_zip_county.columns = nyc_zip_county.columns.str.strip()
nyc_zip_county = nyc_zip_county[['ZIPCODE', 'COUNTY']]

gaz_zip_county = pd.merge(left = gaz_zip, right = nyc_zip_county, left_on = 'GEOID', right_on = 'ZIPCODE', how = 'left')

from sklearn.neighbors import KDTree
kdt = KDTree(gaz_zip[['INTPTLAT', 'INTPTLONG']])



X_train['pickup_zipcode'] = gaz_zip_county.loc[kdt.query(X_train[['pickup_latitude', 'pickup_longitude']].to_numpy(), k=1, return_distance=False).squeeze(), 'GEOID'].values
X_train['pickup_county'] = gaz_zip_county.loc[kdt.query(X_train[['pickup_latitude', 'pickup_longitude']].to_numpy(), k=1, return_distance=False).squeeze(), 'COUNTY'].values
X_train['dropoff_zipcode'] = gaz_zip_county.loc[kdt.query(X_train[['dropoff_latitude', 'dropoff_longitude']].to_numpy(), k=1, return_distance=False).squeeze(), 'GEOID'].values
X_train['dropoff_county'] = gaz_zip_county.loc[kdt.query(X_train[['dropoff_latitude', 'dropoff_longitude']].to_numpy(), k=1, return_distance=False).squeeze(), 'COUNTY'].values
X_train[['pickup_county', 'dropoff_county']] = X_train[['pickup_county', 'dropoff_county']].fillna('not_in_NYC')
X_train['dropoff_pickup_same_county'] = (X_train['dropoff_county'] == X_train['pickup_county'])



X_valid['pickup_zipcode'] = gaz_zip_county.loc[kdt.query(X_valid[['pickup_latitude', 'pickup_longitude']].to_numpy(), k=1, return_distance=False).squeeze(), 'GEOID'].values
X_valid['pickup_county'] = gaz_zip_county.loc[kdt.query(X_valid[['pickup_latitude', 'pickup_longitude']].to_numpy(), k=1, return_distance=False).squeeze(), 'COUNTY'].values
X_valid['dropoff_zipcode'] = gaz_zip_county.loc[kdt.query(X_valid[['dropoff_latitude', 'dropoff_longitude']].to_numpy(), k=1, return_distance=False).squeeze(), 'GEOID'].values
X_valid['dropoff_county'] = gaz_zip_county.loc[kdt.query(X_valid[['dropoff_latitude', 'dropoff_longitude']].to_numpy(), k=1, return_distance=False).squeeze(), 'COUNTY'].values
X_valid[['pickup_county', 'dropoff_county']] = X_valid[['pickup_county', 'dropoff_county']].fillna('not_in_NYC')
X_valid['dropoff_pickup_same_county'] = (X_valid['dropoff_county'] == X_valid['pickup_county'])


#%% airport

# '11430', 07114, 11371
"""# fizzy match of zipcodes; we used KDTree to get the zipcode, which was not exact"""

airport_zipcode = ['11371', '11430', '11414', '11420', '11436', '11434', '11413', '11422', '11581', '11096', '11369', '11370', '11372', '11105', '07114', '07201', '07105', '07208', '07112', '07108', '07102']


X_train['pickup_airport'] = X_train['pickup_zipcode'].isin(airport_zipcode)
X_train['dropoff_airport'] = X_train['dropoff_zipcode'].isin(airport_zipcode)

X_valid['pickup_airport'] = X_valid['pickup_zipcode'].isin(airport_zipcode)
X_valid['dropoff_airport'] = X_valid['dropoff_zipcode'].isin(airport_zipcode)

gc.collect()

# X_train.to_csv('engineered_datasets/X_train.csv')
# X_valid.to_csv('engineered_datasets/X_valid.csv')

#%% one-hot encoding

# !!!
""" one-hot encoding"""

onehot_cols1 = ['passenger_count', 'pickup_county', 'dropoff_county']

from sklearn.preprocessing import OneHotEncoder

# up to certain version, OneHotEncoder could not process string values directly. If your nominal features are strings, then you need to first map them into integers.
OH_encoder1 = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype = 'float32')# sparse=False # will return sparse matrix if set True else will return an array
""" output datatype = 'float32' to save memory"""

def one_hot_fit_transform(X) : 
    
    # transform df using one-hot encoding
    X_object = X[onehot_cols1]
    OH_encoder1.fit(X_object)
    OH_cols = pd.DataFrame(OH_encoder1.transform(X_object), columns = OH_encoder1.get_feature_names(onehot_cols1)) 
    OH_cols.index = X.index
    # drop/keep the pre-encoded columns in case we want to encode them in another way later
    num_X = X.drop(onehot_cols1, axis=1)
    OH_X = pd.concat([num_X, OH_cols], axis=1)
    # OH_X = pd.concat([X, OH_cols], axis=1)
    
    return OH_X

# OH_encoder.transform dataframe
def one_hot_transform(X) : 
    
    # transform df using one-hot encoding
    X_object = X[onehot_cols1]
    OH_cols = pd.DataFrame(OH_encoder1.transform(X_object), columns = OH_encoder1.get_feature_names(onehot_cols1)) 
    OH_cols.index = X.index
    num_X = X.drop(onehot_cols1, axis=1)
    OH_X = pd.concat([num_X, OH_cols], axis=1)
    # OH_X = pd.concat([X, OH_cols], axis=1)
    
    return OH_X


X_train_encoded1 = one_hot_fit_transform(X_train)
X_valid_encoded1 = one_hot_transform(X_valid)

del X_train, X_valid

gc.collect()

#%% frequency-encoding

import category_encoders as ce

freq_cols2 = ['weekday']


# change data type to object before feeding it into the encoder
X_train_encoded1[freq_cols2] = X_train_encoded1[freq_cols2].astype(object)
X_valid_encoded1[freq_cols2] = X_valid_encoded1[freq_cols2].astype(object)

freq_encoder2 = ce.count.CountEncoder()
freq_encoder2.fit(X_train_encoded1[freq_cols2])

X_train_encoded2 = pd.concat([X_train_encoded1, freq_encoder2.transform(X_train_encoded1[freq_cols2]).rename(columns = dict(zip(freq_cols2, [col + '_freq_encoded' for col in freq_cols2])))], axis = 1).drop(freq_cols2, axis = 1)
# downcast to int32 to save ram
X_train_encoded2[[col + '_freq_encoded' for col in freq_cols2]] = X_train_encoded2[[col + '_freq_encoded' for col in freq_cols2]].astype('int32')


X_valid_encoded2 = pd.concat([X_valid_encoded1, freq_encoder2.transform(X_valid_encoded1[freq_cols2]).rename(columns = dict(zip(freq_cols2, [col + '_freq_encoded' for col in freq_cols2])))], axis = 1).drop(freq_cols2, axis = 1)
X_valid_encoded2[[col + '_freq_encoded' for col in freq_cols2]] = X_valid_encoded2[[col + '_freq_encoded' for col in freq_cols2]].astype('int32')

# X_train_encoded2 = X_train_encoded1
# X_valid_encoded2 = X_valid_encoded1

del X_train_encoded1, X_valid_encoded1

#%% target mean encoding, subtracting global mean of the training set

target_mean_cols3 = ['year', 'month', 'day', 'hour', 'pickup_zipcode', 'dropoff_zipcode']

# these columns have been converted to object type for TargetEncoder
target_mean_encoder3 = ce.target_encoder.TargetEncoder()

target_mean_encoder3.fit(X_train_encoded2[target_mean_cols3], y_train)

X_train_encoded3 = pd.concat([X_train_encoded2, target_mean_encoder3.transform(X_train_encoded2[target_mean_cols3]).rename(columns = dict(zip(target_mean_cols3, [col + '_mean_encoded' for col in target_mean_cols3])))], axis = 1).drop(target_mean_cols3, axis = 1)


X_valid_encoded3 = pd.concat([X_valid_encoded2, target_mean_encoder3.transform(X_valid_encoded2[target_mean_cols3]).rename(columns = dict(zip(target_mean_cols3, [col + '_mean_encoded' for col in target_mean_cols3])))], axis = 1).drop(target_mean_cols3, axis = 1)

del X_train_encoded2, X_valid_encoded2

#%% drop and re-order columns
# 'driving_distance',
cols_to_keep4 = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
       'dropoff_latitude', 'daylight_saving', 'long_displacement',
       'lat_displacement', 'euclidean_distance_miles', 
       'sin_direction', 'cos_direction', 'dropoff_pickup_same_county',
       'pickup_airport', 'dropoff_airport', 'passenger_count_1',
       'passenger_count_2', 'passenger_count_3', 'passenger_count_4',
       'passenger_count_5', 'passenger_count_6', 'pickup_county_Bronx',
       'pickup_county_Kings', 'pickup_county_New York', 'pickup_county_Queens',
       'pickup_county_Richmond', 'pickup_county_not_in_NYC',
       'dropoff_county_Bronx', 'dropoff_county_Kings',
       'dropoff_county_New York', 'dropoff_county_Queens',
       'dropoff_county_Richmond', 'dropoff_county_not_in_NYC',
       'weekday_freq_encoded', 'year_mean_encoded', 'month_mean_encoded',
       'day_mean_encoded', 'hour_mean_encoded', 'pickup_zipcode_mean_encoded',
       'dropoff_zipcode_mean_encoded']


X_train_encoded4 = X_train_encoded3[cols_to_keep4]
# X_train_encoded4 = X_train_encoded4.astype(float)
X_valid_encoded4 = X_valid_encoded3[cols_to_keep4]
# X_valid_encoded4 = X_valid_encoded4.astype(float)

del X_train_encoded3, X_valid_encoded3

gc.collect()

#%% change datatype

X_train_encoded4[['passenger_count_1', 'passenger_count_2',
       'passenger_count_3', 'passenger_count_4', 'passenger_count_5',
       'passenger_count_6', 'pickup_county_Bronx', 'pickup_county_Kings',
       'pickup_county_New York', 'pickup_county_Queens',
       'pickup_county_Richmond', 'pickup_county_not_in_NYC',
       'dropoff_county_Bronx', 'dropoff_county_Kings',
       'dropoff_county_New York', 'dropoff_county_Queens',
       'dropoff_county_Richmond', 'dropoff_county_not_in_NYC']] = X_train_encoded4[['passenger_count_1', 'passenger_count_2',
       'passenger_count_3', 'passenger_count_4', 'passenger_count_5',
       'passenger_count_6', 'pickup_county_Bronx', 'pickup_county_Kings',
       'pickup_county_New York', 'pickup_county_Queens',
       'pickup_county_Richmond', 'pickup_county_not_in_NYC',
       'dropoff_county_Bronx', 'dropoff_county_Kings',
       'dropoff_county_New York', 'dropoff_county_Queens',
       'dropoff_county_Richmond', 'dropoff_county_not_in_NYC']].astype('uint8')


#%% fillna with mean

# X_train_encoded4 = X_train_encoded4.fillna(X_train_encoded4.mean())
# X_valid_encoded4 = X_valid_encoded4.fillna(X_valid_encoded4.mean())

#%% scale columns

"""# minmax scaler; not used since 1) we use tree-based algorithms, 2) not enough ram"""

# from sklearn.preprocessing import MinMaxScaler
# scaler5 = MinMaxScaler()

# X_train_encoded4_scaled = scaler5.fit_transform(X_train_encoded4)
# X_train_encoded4_scaled = pd.DataFrame(X_train_encoded4_scaled, columns = X_train_encoded4.columns, index = X_train_encoded4.index)

# X_valid_encoded4_scaled = scaler5.transform(X_valid_encoded4)
# X_valid_encoded4_scaled = pd.DataFrame(X_valid_encoded4_scaled, columns = X_valid_encoded4.columns, index = X_valid_encoded4.index)

# del X_train_encoded4, X_valid_encoded4

#%% transform label to log

# """# use log(y + 1) as label because we use rmsle; using the original label may lead to negative predicted values; error would be nan"""
# y_train_log = np.log(y_train + 1)
# y_valid_log = np.log(y_valid + 1)


#%% save engineered datasets and encoders/scalers

X_train_encoded4.to_csv('engineered_datasets/X_train_encoded4.csv')
X_valid_encoded4.to_csv('engineered_datasets/X_valid_encoded4.csv')
y_train.to_csv('engineered_datasets/y_train.csv')
y_valid.to_csv('engineered_datasets/y_valid.csv')

from pickle import dump


dump(OH_encoder1, open('encoders/OH_encoder1.pkl', 'wb'))
dump(freq_encoder2, open('encoders/freq_encoder2.pkl', 'wb'))
dump(target_mean_encoder3, open('encoders/target_mean_encoder3.pkl', 'wb'))
# dump(scaler5, open('encoders/2021-09-06_5/scaler5.pkl', 'wb'))


#%% load engineered datasets for model training
# !!!

train_cols = pd.read_csv('engineered_datasets/X_train_encoded4.csv', nrows=0).columns.drop(['Unnamed: 0']).tolist()

# downcast datatypes to save RAM
dtypes_new = dict(zip(train_cols, ['float32', 'float32', 'float32', 'float32', bool, 'float32', 'float32', 'float32', 'float32', 'float32', bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32']))

# X_train_encoded4 = dd.read_csv('engineered_datasets/X_train_encoded4.csv', usecols = train_cols, dtype = dtypes_new)
# X_valid_encoded4 = dd.read_csv('engineered_datasets/X_valid_encoded4.csv', usecols = train_cols, dtype = dtypes_new)

""" only load the first 20M rows of training set due to insufficient memory"""
X_train_encoded4 = pd.read_csv('engineered_datasets/X_train_encoded4.csv', index_col = [0], dtype = dtypes_new, nrows = 20000000)
X_valid_encoded4 = pd.read_csv('engineered_datasets/X_valid_encoded4.csv', index_col = [0], dtype = dtypes_new)

# X_train_encoded4_scaled = pd.read_csv('engineered_datasets/2021-09-06_5/X_train_encoded4_scaled.csv', index_col = [0])
# X_valid_encoded4_scaled = pd.read_csv('engineered_datasets/2021-09-06_5/X_valid_encoded4_scaled.csv', index_col = [0])

y_cols = pd.read_csv('engineered_datasets/y_train.csv', nrows=0).columns.drop(['Unnamed: 0']).tolist()

# y_train = dd.read_csv('engineered_datasets/y_train.csv', usecols = y_cols).squeeze()
# y_valid = dd.read_csv('engineered_datasets/y_valid.csv', usecols = y_cols).squeeze()

y_train = pd.read_csv('engineered_datasets/y_train.csv', index_col = [0], nrows = 20000000).squeeze()
y_valid = pd.read_csv('engineered_datasets/y_valid.csv', index_col = [0]).squeeze()

gc.collect()

#%% model scores

from sklearn.metrics import mean_squared_error

"""# DummyRegressor """
# from sklearn.dummy import DummyRegressor
# lm_dummy_mean = DummyRegressor(strategy = 'mean').fit(X_train_encoded4_scaled, y_train)

# print('Dummy Regression: training set R2 score is {:.3f}'.format(lm_dummy_mean.score(X_train_encoded4_scaled, y_train)))
# print('Dummy Regression: validation set R2 score is {:.3f}'.format(lm_dummy_mean.score(X_valid_encoded4_scaled, y_valid)))
# # Dummy Regression: training set R2 score is 0.000
# # Dummy Regression: validation set R2 score is -0.000

# print('Dummy Regression: training set root-mean-squared log error is {:.3f}'.format(np.sqrt(mean_squared_log_error(y_train, lm_dummy_mean.predict(X_train_encoded4_scaled)))))
# print('Dummy Regression: validation set root-mean-squared log error is {:.3f}'.format(np.sqrt(mean_squared_log_error(y_valid, lm_dummy_mean.predict(X_valid_encoded4_scaled)))))
# # Dummy Regression: training set root-mean-squared log error is 0.771
# # Dummy Regression: validation set root-mean-squared log error is 0.771

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

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

# default hyperparameters
XGBR_model = XGBRegressor(eval_metric = "rmse", 
                          learning_rate = 0.05, 
                          max_depth = 8,
                          n_estimators = 100,
                          reg_lambda = 0.7,
                          n_jobs = 6)
XGBR_model.fit(X_train_encoded4, y_train, eval_set = [(X_train_encoded4, y_train), (X_valid_encoded4, y_valid)])

# XGBR_model = XGBRegressor(eval_metric="rmsle", n_jobs = 8)
# XGBR_model.fit(X_train_encoded4_scaled, y_train, eval_set = [(X_train_encoded4_scaled, y_train), (X_valid_encoded4_scaled, y_valid)])


# print(mean_squared_error(y_train, XGBR_model.predict(X_train_encoded4_scaled), squared = False))
# print(mean_squared_error(y_valid, XGBR_model.predict(X_valid_encoded4_scaled), squared = False))

XGBR_feature_importances = pd.Series(XGBR_model.feature_importances_, index = X_train_encoded4.columns).sort_values(ascending = False)
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

XGBR_feature_importances.plot.barh()
plt.title('XGBR')
# plt.savefig('XGBR_feature_importances', dpi = 300)


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

from pickle import dump
# save the model
dump(XGBR_model, open('XGBR_model.pkl', 'wb'))

# my_pred = pd.Series(XGBR_best.predict(encoded_X_test_scaled).clip(0,20), name = 'item_cnt_month', index = test_df_all['ID'])
# # test score 1.69278
# my_pred.to_csv('my_submission_XGBR_best.csv')

# plt.figure()
# plt.scatter(np.array(list(y_train.index)), np.array(list(y_train.values)))
# plt.scatter(np.array(list(y_train.index)), XGBR_best.predict(X_train_encoded4_scaled))

# plt.figure()
# plt.scatter(np.array(list(y_valid.index)), np.array(list(y_valid.values)))
# plt.scatter(np.array(list(y_valid.index)), XGBR_best.predict(X_valid_encoded4_scaled))


#################################

"""# RandomForestRegressor """

from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators = 400, random_state = 0, n_jobs = -1, verbose = 1)
RFR.fit(X_train_encoded4_scaled, y_train)


from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_train, RFR.predict(X_train_encoded4_scaled), squared = False))
print(mean_squared_error(y_valid, RFR.predict(X_valid_encoded4_scaled), squared = False))



pd.Series(RFR.feature_importances_, index = X_train_encoded4_scaled.columns).sort_values(ascending = False).plot.barh()




from pickle import dump
# save the model
dump(RFR, open('RFR_model.pkl', 'wb'))



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

import lightgbm as lgb

LGBMreg = lgb.LGBMRegressor(boosting_type = 'gbdt', 
                            learning_rate = 0.02, 
                            num_leaves = 800,
                            n_estimators = 500, 
                            num_iterations = 5000, 
                            max_bin = 500, 
                            feature_fraction = 0.7, 
                            bagging_fraction = 0.7,
                            lambda_l2 = 0.5,
                            max_depth = 25,
                            silent = False
                            )

LGBMreg.fit(X_train_encoded4, y_train,
            eval_set=[(X_train_encoded4, y_train), (X_valid_encoded4, y_valid)], 
            eval_metric = 'rmse',
            early_stopping_rounds = 50, 
            verbose = True)


# LGBMreg.fit(X_train_encoded4_scaled, y_train,
#             eval_set=[(X_train_encoded4_scaled, y_train), (X_valid_encoded4_scaled, y_valid)],
#             # eval_metric = rmsle, 
#             eval_metric = 'rmse',
#             early_stopping_rounds = 100, 
#             verbose = True)


# print(mean_squared_error(y_train, LGBMreg.predict(X_train_encoded4_scaled), squared = False))
# print(mean_squared_error(y_valid, LGBMreg.predict(X_valid_encoded4_scaled), squared = False))

LGBMreg_feature_importances = pd.Series(LGBMreg.feature_importances_, index = X_train_encoded4.columns).sort_values(ascending = False)
LGBMreg_feature_importances = LGBMreg_feature_importances / LGBMreg_feature_importances.max()

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
# plt.savefig('plots/lightgbm_feature_importances20M.png', dpi = 200)

from pickle import dump
# save the model
dump(LGBMreg, open('LGBMreg.pkl', 'wb'))

#%% test dataset

# !!!

#%% load test dataset
# test_df_all = pd.read_csv('datasets/test.csv', parse_dates = [2], date_parser = parser)

test_df_all_cols = pd.read_csv('engineered_datasets/test_df_datetime.csv', nrows=0).columns
test_df_all_cols = test_df_all_cols.drop(['Unnamed: 0']).tolist()

test_df_all = pd.read_csv(r'engineered_datasets\test_df_datetime.csv', usecols = test_df_all_cols)

X_test = test_df_all.copy()
X_test_id = X_test['key']

# X_test['month'] = X_test['pickup_datetime'].apply(get_month).astype(object)
# X_test['weekday'] = X_test['pickup_datetime'].apply(get_weekday).astype(object)
# X_test['day'] = X_test['pickup_datetime'].apply(get_day).astype(object)
# X_test['hour'] = X_test['pickup_datetime'].apply(get_hour).astype(object)

# pre-rocessing
X_test = euclidean_distance_miles(X_test)

X_test['direction'] = np.arctan2(X_test['lat_displacement'], X_test['long_displacement'])
X_test['sin_direction'] = np.sin(X_test['direction'])
X_test['cos_direction'] = np.cos(X_test['direction'])

X_test['pickup_zipcode'] = gaz_zip_county.loc[kdt.query(X_test[['pickup_latitude', 'pickup_longitude']].to_numpy(), k=1, return_distance=False).squeeze(), 'GEOID'].values
X_test['pickup_county'] = gaz_zip_county.loc[kdt.query(X_test[['pickup_latitude', 'pickup_longitude']].to_numpy(), k=1, return_distance=False).squeeze(), 'COUNTY'].values
X_test['dropoff_zipcode'] = gaz_zip_county.loc[kdt.query(X_test[['dropoff_latitude', 'dropoff_longitude']].to_numpy(), k=1, return_distance=False).squeeze(), 'GEOID'].values
X_test['dropoff_county'] = gaz_zip_county.loc[kdt.query(X_test[['dropoff_latitude', 'dropoff_longitude']].to_numpy(), k=1, return_distance=False).squeeze(), 'COUNTY'].values
X_test[['pickup_county', 'dropoff_county']] = X_test[['pickup_county', 'dropoff_county']].fillna('not_in_NYC')
X_test['dropoff_pickup_same_county'] = (X_test['dropoff_county'] == X_test['pickup_county'])

X_test['pickup_airport'] = X_test['pickup_zipcode'].isin(airport_zipcode)
X_test['dropoff_airport'] = X_test['dropoff_zipcode'].isin(airport_zipcode)

# encoding

from pickle import load
OH_encoder1 = load(open('encoders/OH_encoder1.pkl', 'rb'))
freq_encoder2 = load(open('encoders/freq_encoder2.pkl', 'rb'))
target_mean_encoder3 = load(open('encoders/target_mean_encoder3.pkl', 'rb'))

X_test_encoded1 = one_hot_transform(X_test)
# X_test_encoded2 = X_test_encoded1
X_test_encoded2 = pd.concat([X_test_encoded1, freq_encoder2.transform(X_test_encoded1[freq_cols2]).rename(columns = dict(zip(freq_cols2, [col + '_freq_encoded' for col in freq_cols2])))], axis = 1).drop(freq_cols2, axis = 1)
# X_test_encoded2[[col + '_freq_encoded' for col in freq_cols2]] = X_test_encoded2[[col + '_freq_encoded' for col in freq_cols2]].astype('int32')
X_test_encoded3 = pd.concat([X_test_encoded2, target_mean_encoder3.transform(X_test_encoded2[target_mean_cols3]).rename(columns = dict(zip(target_mean_cols3, [col + '_mean_encoded' for col in target_mean_cols3])))], axis = 1).drop(target_mean_cols3, axis = 1)
X_test_encoded4 = X_test_encoded3[cols_to_keep4]
X_test_encoded4 = X_test_encoded4.astype(float)
# X_test_encoded4 = X_test_encoded4.fillna(X_test_encoded4.mean())

# X_test_encoded4_scaled = scaler5.transform(X_test_encoded4)
# X_test_encoded4_scaled = pd.DataFrame(X_test_encoded4_scaled, columns = X_test_encoded4.columns, index = X_test_encoded4.index)

del X_test_encoded1, X_test_encoded2, X_test_encoded3

# X_test_encoded4.to_csv('engineered_datasets/2021-09-07_7/X_test_encoded4.csv')
# X_test_encoded4_scaled.to_csv('engineered_datasets/2021-09-07_7/X_test_encoded4_scaled.csv')

# predict

y_test_predict = pd.Series(LGBMreg.predict(X_test_encoded4), index= X_test_id, name='fare_amount')

y_test_predict.to_csv('predictions/y_test_predict_LGBMreg.csv')



fig = plt.figure('fare_amount_test_hist', dpi = 150)
y_test_predict.to_frame()['fare_amount'].hist(bins = 100, grid = False, ax = plt.gca(), color = 'tomato')
plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Fare amount in USD')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0.1,)
# plt.savefig('plots/fare_amount_pred_hist.png', dpi = 150)

