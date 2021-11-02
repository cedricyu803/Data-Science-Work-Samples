# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:50:00 2021

@author: Cedric Yu
"""

# import matplotlib.pyplot as plt
# import math
# import re


"""
# Assignment 4 - Understanding and Predicting Property Maintenance Fines

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

#%% Preamble

import pandas as pd
# Make the output look better
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt


import os
os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\2_detroit_blight_ticket_classification')


#%% training datasets


# list(train_df.columns)

# ['ticket_id',   # NOT a feature!
#  'agency_name',            # 5 names includes test instances
#  'inspector_name'          # 159 names, some test instances not included

#  'violator_name',          # 84657 names
#  'violation_street_number',   # 18096
#  'violation_street_name',  # 1716
#  'violation_zip_code', # all nan, 59 for test set

#  'mailing_address_str_number',    # 14091
#  'mailing_address_str_name',  # 28441
#  'city', # 4093
#  'state', # 60 with nan
#  'zip_code', # there are 9XXXX-XXXX and nan, 817 has less than 5 digits
#  'non_us_str_code', 
#  'country', # 5 countries, 11 non-US, test set all in US

#  'ticket_issued_date', # no nan
#  'hearing_date', # 227 nan
#  'violation_code', # 189, 151 for test
#  'violation_description', # mapped to violation_code
#  'disposition', # 4 objects, 8 for test
#  'fine_amount',  # 40, 53 for test
#  'admin_fee',  # all 20.
#  'state_fee',  # all 10.
#  'late_fee',    # 37, 44 for test
#  'discount_amount',  # 13, 14 for test
#  'clean_up_cost', # all 0. (KEEP??? ; test data have 298 unique values)
#  'judgment_amount', # 57       # it is zero if 'fine_amount' is zero, and is otherwise the sum of above fees, not including discount
#  'grafitti_status'] all nan


"""
Observations
# ticket_id should not be taken as a feature to avoid data leakage, as it may contain information that has to do with the target
# few agency names; use one-hot encoding, i.e. get_dummies
# many inspectors
# many violators and their addresses
# many mailing addresses, few (11) outside the US
# Many mis-categorised countries into 'USA'
# zip_code needs processing
# hearing_date has some nan, has similar std to ticket_issue_date Their different has rather small std
# violation_description is redundant
# admin_fee and state_fee are all the same
# clean_up_cost are all zero, but test data has a lot of non-zeros
# judgment_amount is a simple sum of fees
# graffiti_status is all nan, but not so for test set

To do:
# Do the following separately for train and test sets to avoid leakage!

# Drop 
'ticket_id', 'violation_zip_code', 'violation_description', 'admin_fee', 'state_fee', 'judgment_amount', 'grafitti_status'
# Drop? 
'inspector_name', 'violator_name', 'violation_street_number', 'violation_street_name', 'mailing_address_str_number', 'mailing_address_str_name', 'city', 'state', 'non_us_str_code'
# Drop?
'hearing_date', 'clean_up_cost'

# In 'country', assign US mailing addresses as 1, otherwise 0
# convert legit zip_code to 5-digit integers, and set anything else (less than 5 digits and non-US zip codes) to 0
# drop 'country' improves ROC-AUC scores generally
# convert 'ticket_issued_date' to seconds since epoch
# use pd.get_dummies or one-hot encoding to convert the categorical variables into indicator variables. Use set.transform on training set and transform on test set to avoid leakage!!
# 'agency_name', 'violation_code', 'disposition'
"""
#%% define functions for pre-processing
#%% drop lesss-useful columns from X

def drop_cols(df) : 
    cols_to_drop=['ticket_id', 'violation_zip_code', 'violation_description', 'admin_fee', 'state_fee', 'judgment_amount', 'inspector_name', 'violator_name', 'violation_street_number', 'violation_street_name', 'mailing_address_str_number', 'mailing_address_str_name', 'non_us_str_code', 'city', 'hearing_date', 'clean_up_cost', 'grafitti_status']
    return df.drop(cols_to_drop, axis=1)

#%% processing 'country' and 'zip_code', and drop 'state'

"""
# If 'state' is really one of the 51 states (including DC), and zip_code is 5 legit digits or more, take it as really being in the US and assign 1 to 'country', and we set the zip code using regex,
# otherwise 'country' and 'zip_code' are set to 0
"""

def country_zip_process(df) :
    
    US_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
              "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
              "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
              "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
              "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
    
    def country_zip_func(row) : 
        import re
        if (row['state'] in US_states) & (len(re.findall("\d{5,5}",row['zip_code'])) > 0) : 
            row['country']=1
            row['zip_code']=int(re.findall("\d{5,5}",row['zip_code'])[0])
        else : 
            row['country']=0
            row['zip_code']=0
        return row
    
    # convert zip_code to str
    df['zip_code']=df['zip_code'].astype(str)
    df1 = df.apply(country_zip_func, axis='columns')
    
    return df1.drop('state', axis=1).drop('country',axis=1)


#%% convert 'ticket_issued_date' to seconds since epoch

def ticketdatetosec(df) : 
    
    from datetime import datetime
    import time
    
    def ticketdatetosec_func(row):
        row['ticket_issued_date']=time.mktime(datetime.strptime(row['ticket_issued_date'], "%Y-%m-%d %H:%M:%S").timetuple())
        return row
    
    return df.apply(ticketdatetosec_func, axis='columns')

# def hearingdatetosec(row):
#     if pd.isnull(row['hearing_date']) : 
#         row['hearing_date'] = -1
#     else : 
#         row['hearing_date']=time.mktime(datetime.strptime(row['hearing_date'], "%Y-%m-%d %H:%M:%S").timetuple())
#     return row

# train_df=train_df.apply(hearingdatetosec, axis='columns')
# train_df['hearing_date']=train_df['hearing_date'].replace(-1.0, np.nan)

#%% The remaining categorical variables--- one-hot encoding
"""
violation_code has many distinct elements
"""

from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)# sparse=False # will return sparse matrix if set True else will return an array

object_cols=['agency_name', 'violation_code', 'disposition']

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

#%% pre-processing summary

def train_preprocess(df) : 
    return one_hot_fit_transform(ticketdatetosec(country_zip_process(df)))

def not_train_preprocess(df) : 
    return one_hot_transform(ticketdatetosec(country_zip_process(df)))


#%% re-scale parameters with MinMaxScaler()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

def train_scaler(df) : return scaler.fit_transform(df)
def non_train_scaler(df) : return scaler.transform(df)







