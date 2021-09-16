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
import gc

import dask
import dask.dataframe as dd


import os
os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\7_new-york-city-taxi-fare-prediction')


#%% load dataset datetime-processed train.csv
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

#%% functions and column selections

"""# returns long_displacement, lat_displacement and euclidean_distance_miles"""

def euclidean_distance_miles(X):
    
    X_ = X.copy()
    R_earth = 1.  # radius of the earth in miles = 3959. feature will be rescaled, so set it to 1
    
    X_['long_displacement'] = (R_earth * ((X_['dropoff_longitude'] - X_['pickup_longitude']) * np.pi / 180) * np.cos( 0.5*(X_['dropoff_latitude'] + X_['pickup_latitude']) * np.pi / 180 )).astype('float32')
    X_['lat_displacement'] = (R_earth * (X_['dropoff_latitude'] - X_['pickup_latitude']) * np.pi / 180).astype('float32')
    X_['euclidean_distance_miles'] = (np.sqrt( X_['long_displacement'] * X_['long_displacement'] + X_['lat_displacement'] * X_['lat_displacement'] )).astype('float32')
    
    return X_


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


""" airport zipcodes"""
# '11430', 07114, 11371
"""# fizzy match of zipcodes; we used KDTree to get the zipcode, which was not exact"""

airport_zipcode = ['11371', '11430', '11414', '11420', '11436', '11434', '11413', '11422', '11581', '11096', '11369', '11370', '11372', '11105', '07114', '07201', '07105', '07208', '07112', '07108', '07102']


"""encoding"""

onehot_cols1 = ['passenger_count', 'pickup_county', 'dropoff_county']
freq_cols2 = ['weekday']
target_mean_cols3 = ['year', 'month', 'day', 'hour', 'pickup_zipcode', 'dropoff_zipcode']

from sklearn.preprocessing import OneHotEncoder

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

import category_encoders as ce


"""columns to keep """
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


cols_to_downcast = ['passenger_count_1', 'passenger_count_2',
       'passenger_count_3', 'passenger_count_4', 'passenger_count_5',
       'passenger_count_6', 'pickup_county_Bronx', 'pickup_county_Kings',
       'pickup_county_New York', 'pickup_county_Queens',
       'pickup_county_Richmond', 'pickup_county_not_in_NYC',
       'dropoff_county_Bronx', 'dropoff_county_Kings',
       'dropoff_county_New York', 'dropoff_county_Queens',
       'dropoff_county_Richmond', 'dropoff_county_not_in_NYC']

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



#%% train+validation set pre-processing

# euclidean distances and bearing
X_train = euclidean_distance_miles(X_train)
X_valid = euclidean_distance_miles(X_valid)

# direction of travel
X_train['direction'] = np.arctan2(X_train['lat_displacement'], X_train['long_displacement']).astype('float32')
X_train['sin_direction'] = np.sin(X_train['direction']).astype('float32')
X_train['cos_direction'] = np.cos(X_train['direction']).astype('float32')
X_train.drop(['direction'], axis = 1, inplace = True)

X_valid['direction'] = np.arctan2(X_valid['lat_displacement'], X_valid['long_displacement']).astype('float32')
X_valid['sin_direction'] = np.sin(X_valid['direction']).astype('float32')
X_valid['cos_direction'] = np.cos(X_valid['direction']).astype('float32')
X_valid.drop(['direction'], axis = 1, inplace = True)


# zip codes and boroughs
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


# airport
X_train['pickup_airport'] = X_train['pickup_zipcode'].isin(airport_zipcode)
X_train['dropoff_airport'] = X_train['dropoff_zipcode'].isin(airport_zipcode)

X_valid['pickup_airport'] = X_valid['pickup_zipcode'].isin(airport_zipcode)
X_valid['dropoff_airport'] = X_valid['dropoff_zipcode'].isin(airport_zipcode)


# one-hot encoding
# up to certain version, OneHotEncoder could not process string values directly. If your nominal features are strings, then you need to first map them into integers.
OH_encoder1 = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype = 'float32')# sparse=False # will return sparse matrix if set True else will return an array
# output datatype = 'float32' to save memory

X_train_encoded1 = one_hot_fit_transform(X_train)
X_valid_encoded1 = one_hot_transform(X_valid)

del X_train, X_valid

# frequency-encoding
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

# use this in case we do not use frequency encoding
# X_train_encoded2 = X_train_encoded1
# X_valid_encoded2 = X_valid_encoded1

del X_train_encoded1, X_valid_encoded1

# target mean encoding
# these columns have been converted to object type for TargetEncoder
target_mean_encoder3 = ce.target_encoder.TargetEncoder()
target_mean_encoder3.fit(X_train_encoded2[target_mean_cols3], y_train)

X_train_encoded3 = pd.concat([X_train_encoded2, target_mean_encoder3.transform(X_train_encoded2[target_mean_cols3]).rename(columns = dict(zip(target_mean_cols3, [col + '_mean_encoded' for col in target_mean_cols3])))], axis = 1).drop(target_mean_cols3, axis = 1)


X_valid_encoded3 = pd.concat([X_valid_encoded2, target_mean_encoder3.transform(X_valid_encoded2[target_mean_cols3]).rename(columns = dict(zip(target_mean_cols3, [col + '_mean_encoded' for col in target_mean_cols3])))], axis = 1).drop(target_mean_cols3, axis = 1)

del X_train_encoded2, X_valid_encoded2

# drop and re-order columns
X_train_encoded4 = X_train_encoded3[cols_to_keep4]
# X_train_encoded4 = X_train_encoded4.astype(float)
X_valid_encoded4 = X_valid_encoded3[cols_to_keep4]
# X_valid_encoded4 = X_valid_encoded4.astype(float)

del X_train_encoded3, X_valid_encoded3


# downcast datatypes
X_train_encoded4[cols_to_downcast] = X_train_encoded4[cols_to_downcast].astype('uint8')


# fillna with mean
# X_train_encoded4 = X_train_encoded4.fillna(X_train_encoded4.mean())
# X_valid_encoded4 = X_valid_encoded4.fillna(X_valid_encoded4.mean())

# scale columns

# minmax scaler; not used since 1) we use tree-based algorithms, 2) not enough ram
# from sklearn.preprocessing import MinMaxScaler
# scaler5 = MinMaxScaler()

# X_train_encoded4_scaled = scaler5.fit_transform(X_train_encoded4)
# X_train_encoded4_scaled = pd.DataFrame(X_train_encoded4_scaled, columns = X_train_encoded4.columns, index = X_train_encoded4.index)

# X_valid_encoded4_scaled = scaler5.transform(X_valid_encoded4)
# X_valid_encoded4_scaled = pd.DataFrame(X_valid_encoded4_scaled, columns = X_valid_encoded4.columns, index = X_valid_encoded4.index)

# del X_train_encoded4, X_valid_encoded4


#%% save engineered datasets and encoders/scalers

X_train_encoded4.to_csv('engineered_datasets/X_train_encoded4.csv')
X_valid_encoded4.to_csv('engineered_datasets/X_valid_encoded4.csv')
y_train.to_csv('engineered_datasets/y_train.csv')
y_valid.to_csv('engineered_datasets/y_valid.csv')


# save encoders for test set pre-processing 
from pickle import dump

dump(OH_encoder1, open('encoders/OH_encoder1.pkl', 'wb'))
dump(freq_encoder2, open('encoders/freq_encoder2.pkl', 'wb'))
dump(target_mean_encoder3, open('encoders/target_mean_encoder3.pkl', 'wb'))
# dump(scaler5, open('encoders/scaler5.pkl', 'wb'))


#%% test dataset

#%% load test dataset
# test_df_all = pd.read_csv('datasets/test.csv', parse_dates = [2], date_parser = parser)

test_df_all_cols = pd.read_csv('engineered_datasets/test_df_datetime.csv', nrows=0).columns
test_df_all_cols = test_df_all_cols.drop(['Unnamed: 0']).tolist()

test_df_all = pd.read_csv(r'engineered_datasets\test_df_datetime.csv', usecols = test_df_all_cols)

X_test = test_df_all.copy()
X_test_id = X_test['key']

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
# load fitted encoders
from pickle import load
OH_encoder1 = load(open('encoders/OH_encoder1.pkl', 'rb'))
freq_encoder2 = load(open('encoders/freq_encoder2.pkl', 'rb'))
target_mean_encoder3 = load(open('encoders/target_mean_encoder3.pkl', 'rb'))

X_test_encoded1 = one_hot_transform(X_test)
# X_test_encoded2 = X_test_encoded1
X_test_encoded2 = pd.concat([X_test_encoded1, freq_encoder2.transform(X_test_encoded1[freq_cols2]).rename(columns = dict(zip(freq_cols2, [col + '_freq_encoded' for col in freq_cols2])))], axis = 1).drop(freq_cols2, axis = 1)
X_test_encoded3 = pd.concat([X_test_encoded2, target_mean_encoder3.transform(X_test_encoded2[target_mean_cols3]).rename(columns = dict(zip(target_mean_cols3, [col + '_mean_encoded' for col in target_mean_cols3])))], axis = 1).drop(target_mean_cols3, axis = 1)
X_test_encoded4 = X_test_encoded3[cols_to_keep4]
# X_test_encoded4 = X_test_encoded4.fillna(X_test_encoded4.mean())

# X_test_encoded4_scaled = scaler5.transform(X_test_encoded4)
# X_test_encoded4_scaled = pd.DataFrame(X_test_encoded4_scaled, columns = X_test_encoded4.columns, index = X_test_encoded4.index)

del X_test_encoded1, X_test_encoded2, X_test_encoded3

# include 'key' column
X_test_encoded4 = pd.concat([X_test_id, X_test_encoded4], axis = 1)
X_test_encoded4.to_csv('engineered_datasets/X_test_encoded4.csv')
# X_test_encoded4_scaled.to_csv('engineered_datasets/2021-09-07_7/X_test_encoded4_scaled.csv')












