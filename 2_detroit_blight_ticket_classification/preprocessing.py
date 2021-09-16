# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 11:00:00 2021

@author: Cedric Yu
"""

# import matplotlib.pyplot as plt
# import math
# import re

# Your AUC of 0.774448781327 was awarded a value of 1.0 out of 1.0 total grades

#%%
"""
This assignment is based on a data challenge from the Michigan Data Science Team (MDST).

The Michigan Data Science Team (MDST) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences (MSSISS) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. Blight violations are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?

The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. 

For this assignment, your task is to <<predict whether a given blight ticket will be paid on time.>>

# All data for this assignment has been provided to us through the Detroit Open Data Portal. Only the data already included in your Coursera directory can be used for training the model for this assignment. Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# Building Permits
# Trades Permits
# Improve Detroit: Submitted Issues
# DPD: Citizen Complaints
# Parcel Map


We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.

Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.


File descriptions (Use only this data for training your model!)

readonly/train.csv - the training set (all tickets issued 2004-2011)
readonly/test.csv - the test set (all tickets issued 2012-2016)
readonly/addresses.csv & readonly/latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
 Note: misspelled addresses may be incorrectly geolocated.

Data fields

train.csv & test.csv

ticket_id - unique identifier for tickets
agency_name - Agency that issued the ticket
inspector_name - Name of inspector that issued the ticket
violator_name - Name of the person/organization that the ticket was issued to
violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
ticket_issued_date - Date and time the ticket was issued
hearing_date - Date and time the violator's hearing was scheduled
violation_code, violation_description - Type of violation
disposition - Judgment and judgement type
fine_amount - Violation fine amount, excluding fees
admin_fee - $20 fee assigned to responsible judgments
state_fee - $10 fee assigned to responsible judgments
late_fee - 10% fee assigned to responsible judgments
discount_amount - discount applied, if any
clean_up_cost - DPW clean-up or graffiti removal cost
judgment_amount - Sum of all fines and fees
grafitti_status - Flag for graffiti violations

train.csv only

payment_amount - Amount paid, if any
payment_date - Date payment was made, if it was received
payment_status - Current payment status as of Feb 1 2017
balance_due - Fines and fees still owed
collection_status - Flag for payments in collections
compliance [target variable for prediction] 
 Null = Not responsible
 0 = Responsible, non-compliant
 1 = Responsible, compliant
compliance_detail - More information on why each ticket was marked compliant or non-compliant


# Evaluation
Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.

<<The evaluation metric for this assignment is the Area Under the ROC Curve (AUC).>>

<<Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.>>

For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using readonly/train.csv. 
Using this model, <<return a series of length 61001 with the data being the <<probability>> that each corresponding ticket from readonly/test.csv will be paid, and the index being the ticket_id.>>

Example:

ticket_id
   284932    0.531842
   285362    0.401958
   285361    0.105928
   285338    0.018572
             ...
   376499    0.208567
   376500    0.818759
   369851    0.018528
   Name: compliance, dtype: float32

Hints
Make sure your code is working before submitting it to the autograder.

Print out your result to see whether there is anything weird (e.g., all probabilities are the same).

Generally the total runtime should be less than 10 mins. You should NOT use Neural Network related classifiers (e.g., MLPClassifier) in this question.

Try to avoid global variables. If you have other functions besides blight_model, you should move those functions inside the scope of blight_model.

Refer to the pinned threads in Week 4's discussion forum when there is something you could not figure it out. 

"""



#%% Workflow

"""
# load datetime-processed training dataset
# drop columns that are not in the test set
# drop instances with nan target label ('compliance')
# restrict to instances after 2004
# fill (the one) nan of 'zip_code'
# drop columns: 
#     cols_to_drop1 = ['ticket_id', 'violation_zip_code', 'violation_description', 'admin_fee', 'state_fee', 'judgment_amount', 'inspector_name', 'violator_name', 'violation_street_number', 'violation_street_name', 'mailing_address_str_number', 'mailing_address_str_name', 'non_us_str_code', 'city', 'clean_up_cost', 'grafitti_status', 'ticket_issued_date_year', 'hearing_date_year']
# separate features and target
# train-validation split (0.8-0.2)
# process mailing address:
#     If 'state' is really one of the 51 states (including DC), take first 5 digits of zip_code if possible (else set 0), take it as really being in the US and assign 1 to 'country'
#     otherwise 'country' and 'zip_code' are set to 0, 'state' set to 'not_in_US'
# fillna for hearing date columns with most frequent occurrence
# encoding
# fillna in validation set that resulted from encoding
# scaler



"""

#%% Preamble

import pandas as pd
# Make the output look better
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import gc

import os
os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\2_detroit_blight_ticket_classification')


#%% load datetime-processed training set


# get training set column names
train_cols_new = pd.read_csv('engineered_datasets/train_df_datetime.csv', nrows=0).columns
train_cols_new = train_cols_new.drop('Unnamed: 0').to_list()

# import datetime-processed training dataset
train_df_raw = pd.read_csv('engineered_datasets/train_df_datetime.csv', low_memory = True, usecols = train_cols_new, dtype = {'zip_code': str, 'non_us_str_code': str, 'grafitti_status': str, 'ticket_issued_date_month': int, 'ticket_issued_date_weekday': int, 'ticket_issued_date_day': int, 'ticket_issued_date_hour': int})


train_df = train_df_raw.copy()

#%% drop some columns and rows

train_df = train_df[~np.isnan(train_df['compliance'])] 
# drop instances with NaN compliance

# drop columns that are only accessible when compliance is made
train_df.drop(['payment_amount','payment_date','payment_status','balance_due','collection_status','compliance_detail'], inplace=True,axis=1)

train_df = train_df[train_df['ticket_issued_date_year'] > 2004]
train_df['zip_code'] = train_df['zip_code'].fillna('nfd')

train_df.reset_index(inplace=True)
train_df.drop('index', inplace=True,axis=1)

#%% drop columns

cols_to_drop1 = ['ticket_id', 'violation_zip_code', 'violation_description', 'admin_fee', 'state_fee', 'judgment_amount', 'inspector_name', 'violator_name', 'violation_street_number', 'violation_street_name', 'mailing_address_str_number', 'mailing_address_str_name', 'non_us_str_code', 'city', 'clean_up_cost', 'grafitti_status', 'ticket_issued_date_year', 'hearing_date_year']


train_df = train_df.drop(cols_to_drop1, axis = 1)

#%% 

X_train_valid = train_df.drop('compliance', axis = 1)
y_train_valid = train_df['compliance']

# del train_df

#%% train-validation split

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, train_size = 0.8)

# del X_train_valid, y_train_valid

gc.collect()

#%% functions and column selections


"""
# If 'state' is really one of the 51 states (including DC), take first 5 digits of zip_code if possible (else set 0), take it as really being in the US and assign 1 to 'country'.
# otherwise 'country' and 'zip_code' are set to 0, state set to 'not_in_US'
"""

def country_zip_process(df) :
    
    US_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
              "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
              "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
              "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
              "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
    
    def country_zip_func(row) : 
        if row['state'] in US_states: 
            row['country'] = 1
            if (len(re.findall("\d{5,5}",row['zip_code'])) > 0):
                row['zip_code']=int(re.findall("\d{5,5}",row['zip_code'])[0])
            else:
                row['zip_code'] = 0
        else : 
            row['country'] = 0
            row['state'] = 'not_in_US'
            row['zip_code'] = 0
        return row
    
    # convert zip_code to str
    # df['zip_code'] = df['zip_code'].astype(str)
    df1 = df.apply(country_zip_func, axis = 1)
    
    return df1


"""# fill na in hearing datetime with most frequenct"""
month_fill = 4
weekday_fill = 1
day_fill = 20
hour_fill = 9


"""encoding """

object_cols=['agency_name', 'disposition']

freq_cols2 = ['violation_code', 'hearing_date_day', 'ticket_issued_date_weekday', 'hearing_date_weekday', 'hearing_date_hour', 'state']

target_mean_cols3 = ['ticket_issued_date_month', 'hearing_date_month', 'ticket_issued_date_day', 'ticket_issued_date_hour']



"""One-hot """
from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)# sparse=False # will return sparse matrix if set True else will return an array

# OH_encoder.fit_transform dataframe
def one_hot_fit_transform(df) : 
    
    # transform df using one-hot encoding
    df_object = df[object_cols]
    OH_encoder.fit(df_object)
    OH_cols = pd.DataFrame(OH_encoder.transform(df_object), columns = OH_encoder.get_feature_names(object_cols)) 
    OH_cols.index = df.index
    num_X = df.drop(object_cols, axis=1)
    OH_X = pd.concat([num_X, OH_cols], axis=1)
    
    return OH_X

# OH_encoder.transform dataframe
def one_hot_transform(df) : 
    
    # transform df using one-hot encoding
    df_object = df[object_cols]
    OH_cols = pd.DataFrame(OH_encoder.transform(df_object), columns = OH_encoder.get_feature_names(object_cols)) 
    OH_cols.index = df.index
    num_X = df.drop(object_cols, axis=1)
    OH_X = pd.concat([num_X, OH_cols], axis=1)
    
    return OH_X


"""for frequency and mean encoding """
import category_encoders as ce

"""scaler """
from sklearn.preprocessing import MinMaxScaler
scaler5 = MinMaxScaler()




#%% pre-processing

"""training set """
"""# processing 'country', 'state' and 'zip_code' """
X_train = country_zip_process(X_train)

"""# fillna in hearing datetime"""
X_train['hearing_date_month'] = X_train['hearing_date_month'].fillna(month_fill).astype(int)
X_train['hearing_date_weekday'] = X_train['hearing_date_weekday'].fillna(weekday_fill).astype(int)
X_train['hearing_date_day'] = X_train['hearing_date_day'].fillna(day_fill).astype(int)
X_train['hearing_date_hour'] = X_train['hearing_date_hour'].fillna(hour_fill).astype(int)

"""# one-hot encoding"""
X_train_encoded1 = one_hot_fit_transform(X_train)

"""# frequency encoding"""
# change data type to object before feeding it into the encoder
X_train_encoded1[freq_cols2] = X_train_encoded1[freq_cols2].astype(object)
freq_encoder2 = ce.count.CountEncoder()
freq_encoder2.fit(X_train_encoded1[freq_cols2])

X_train_encoded2 = pd.concat([X_train_encoded1, freq_encoder2.transform(X_train_encoded1[freq_cols2]).rename(columns = dict(zip(freq_cols2, [col + '_freq_encoded' for col in freq_cols2])))], axis = 1).drop(freq_cols2, axis = 1)
# downcast to int32 to save ram
X_train_encoded2[[col + '_freq_encoded' for col in freq_cols2]] = X_train_encoded2[[col + '_freq_encoded' for col in freq_cols2]]

"""target encoding """
# change data type to object before feeding it into the encoder
X_train_encoded2[target_mean_cols3] = X_train_encoded2[target_mean_cols3].astype(object)

target_mean_encoder3 = ce.target_encoder.TargetEncoder()
target_mean_encoder3.fit(X_train_encoded2[target_mean_cols3], y_train)

X_train_encoded3 = pd.concat([X_train_encoded2, target_mean_encoder3.transform(X_train_encoded2[target_mean_cols3]).rename(columns = dict(zip(target_mean_cols3, [col + '_mean_encoded' for col in target_mean_cols3])))], axis = 1).drop(target_mean_cols3, axis = 1)

"""# fillna resulting from encoding"""
X_train_encoded3 = X_train_encoded3.fillna(X_train_encoded3.mean())

"""scaler """
X_train_encoded4_scaled = scaler5.fit_transform(X_train_encoded3)
X_train_encoded4_scaled = pd.DataFrame(X_train_encoded4_scaled, columns = X_train_encoded3.columns, index = X_train_encoded3.index)



"""validation set """

X_valid = country_zip_process(X_valid)


X_valid['hearing_date_month'] = X_valid['hearing_date_month'].fillna(month_fill).astype(int)
X_valid['hearing_date_weekday'] = X_valid['hearing_date_weekday'].fillna(weekday_fill).astype(int)
X_valid['hearing_date_day'] = X_valid['hearing_date_day'].fillna(day_fill).astype(int)
X_valid['hearing_date_hour'] = X_valid['hearing_date_hour'].fillna(hour_fill).astype(int)


X_valid_encoded1 = one_hot_transform(X_valid)


X_valid_encoded1[freq_cols2] = X_valid_encoded1[freq_cols2].astype(object)
X_valid_encoded2 = pd.concat([X_valid_encoded1, freq_encoder2.transform(X_valid_encoded1[freq_cols2]).rename(columns = dict(zip(freq_cols2, [col + '_freq_encoded' for col in freq_cols2])))], axis = 1).drop(freq_cols2, axis = 1)
X_valid_encoded2[[col + '_freq_encoded' for col in freq_cols2]] = X_valid_encoded2[[col + '_freq_encoded' for col in freq_cols2]]


X_valid_encoded2[target_mean_cols3] = X_valid_encoded2[target_mean_cols3].astype(object)
X_valid_encoded3 = pd.concat([X_valid_encoded2, target_mean_encoder3.transform(X_valid_encoded2[target_mean_cols3]).rename(columns = dict(zip(target_mean_cols3, [col + '_mean_encoded' for col in target_mean_cols3])))], axis = 1).drop(target_mean_cols3, axis = 1)


X_valid_encoded3 = X_valid_encoded3.fillna(X_train_encoded3.mean())


X_valid_encoded4_scaled = scaler5.transform(X_valid_encoded3)
X_valid_encoded4_scaled = pd.DataFrame(X_valid_encoded4_scaled, columns = X_valid_encoded3.columns, index = X_valid_encoded3.index)


#%% load datetime-processed test set


test_cols_new = pd.read_csv('engineered_datasets/test_df_datetime.csv', nrows=0).columns
test_cols_new = test_cols_new.drop('Unnamed: 0').to_list()

# import datetime-processed training dataset
test_df_raw = pd.read_csv('engineered_datasets/test_df_datetime.csv', low_memory = True, usecols = test_cols_new, dtype = {'zip_code': str, 'non_us_str_code': str, 'grafitti_status': str, 'ticket_issued_date_month': int, 'ticket_issued_date_weekday': int, 'ticket_issued_date_day': int, 'ticket_issued_date_hour': int})

test_id = test_df_raw['ticket_id'].copy()
X_test = test_df_raw.copy()


"""test set """
# replace the part for validation set by test set

X_test = X_test.drop(cols_to_drop1, axis = 1)

X_test = country_zip_process(X_test)


X_test['hearing_date_month'] = X_test['hearing_date_month'].fillna(month_fill).astype(int)
X_test['hearing_date_weekday'] = X_test['hearing_date_weekday'].fillna(weekday_fill).astype(int)
X_test['hearing_date_day'] = X_test['hearing_date_day'].fillna(day_fill).astype(int)
X_test['hearing_date_hour'] = X_test['hearing_date_hour'].fillna(hour_fill).astype(int)


X_test_encoded1 = one_hot_transform(X_test)


X_test_encoded1[freq_cols2] = X_test_encoded1[freq_cols2].astype(object)
X_test_encoded2 = pd.concat([X_test_encoded1, freq_encoder2.transform(X_test_encoded1[freq_cols2]).rename(columns = dict(zip(freq_cols2, [col + '_freq_encoded' for col in freq_cols2])))], axis = 1).drop(freq_cols2, axis = 1)
X_test_encoded2[[col + '_freq_encoded' for col in freq_cols2]] = X_test_encoded2[[col + '_freq_encoded' for col in freq_cols2]]


X_test_encoded2[target_mean_cols3] = X_test_encoded2[target_mean_cols3].astype(object)
X_test_encoded3 = pd.concat([X_test_encoded2, target_mean_encoder3.transform(X_test_encoded2[target_mean_cols3]).rename(columns = dict(zip(target_mean_cols3, [col + '_mean_encoded' for col in target_mean_cols3])))], axis = 1).drop(target_mean_cols3, axis = 1)


X_test_encoded3 = X_test_encoded3.fillna(X_train_encoded3.mean())


X_test_encoded4_scaled = scaler5.transform(X_test_encoded3)
X_test_encoded4_scaled = pd.DataFrame(X_test_encoded4_scaled, columns = X_test_encoded3.columns, index = X_test_encoded3.index)


# cor = pd.concat([X_train_encoded3, y_train], axis = 1).corr()
# mask = np.array(cor)
# mask[np.tril_indices_from(mask)] = False
# fig, ax = plt.subplots(dpi = 200)
# fig.set_size_inches(20,30)
# sns.heatmap(cor,mask= mask,square=True, annot = True, annot_kws = {'size': 'xx-small'})
# ax = plt.gca()
# ax.set_xticklabels(ax.get_xticklabels(), fontsize = 'xx-small', rotation = 40)
# # plt.savefig('plots/training dataset/correlation_matrix.png', dpi = 200)


# del X_train_encoded3, X_train_encoded2, X_train_encoded1, X_train
# del X_valid_encoded3, X_valid_encoded2, X_valid_encoded1, X_valid

#%% save pre-processed datasets

X_train_encoded3.to_csv('engineered_datasets/X_train_encoded3.csv')
X_valid_encoded3.to_csv('engineered_datasets/X_valid_encoded3.csv')
X_train_encoded4_scaled.to_csv('engineered_datasets/X_train_encoded4_scaled.csv')
X_valid_encoded4_scaled.to_csv('engineered_datasets/X_valid_encoded4_scaled.csv')
y_train.to_csv('engineered_datasets/y_train.csv')
y_valid.to_csv('engineered_datasets/y_valid.csv')

X_test_encoded4_scaled.to_csv('engineered_datasets/X_test_encoded4_scaled.csv')


