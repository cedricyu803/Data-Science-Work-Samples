# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 11:00:00 2021

@author: Cedric Yu
"""

# import matplotlib.pyplot as plt
# import math
# import re


"""
#####################################

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

#####################################

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


# %% Preamble

# Make the output look better
from datetime import datetime
import os
import seaborn as sns
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# default='warn' # ignores warning about dropping columns inplace
pd.options.mode.chained_assignment = None

os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\2_detroit_blight_ticket_classification')


# !!!
# %% load training and test set

# parse datetime columns
def parser(c): return pd.to_datetime(
    c, format="%Y-%m-%d %H:%M:%S", errors='coerce')
# coerce: invalid parsing will be set as NaT.


# import original training dataset
# train_cols = pd.read_csv(r'train.csv', nrows = 0).columns.tolist()
train_df_all = pd.read_csv('train.csv', encoding='ISO-8859-1',
                           low_memory=False, parse_dates=[14, 15], date_parser=parser)
train_df_all.reset_index(inplace=True)
train_df_all.drop('index', inplace=True, axis=1)

# train_df_raw.shape
train_df = train_df_all.copy()

# pd.read_csv(r'test.csv', nrows = 0).columns.tolist()
test_df_all = pd.read_csv(
    'test.csv', encoding='ISO-8859-1', parse_dates=[14, 15], date_parser=parser)
test_df = test_df_all.copy()


# %% datetime features


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


# !!!
train_df['ticket_issued_date_year'] = train_df['ticket_issued_date'].apply(
    get_year)
train_df['ticket_issued_date_month'] = train_df['ticket_issued_date'].apply(
    get_month)
train_df['ticket_issued_date_weekday'] = train_df['ticket_issued_date'].apply(
    get_weekday)
train_df['ticket_issued_date_day'] = train_df['ticket_issued_date'].apply(
    get_day)
train_df['ticket_issued_date_hour'] = train_df['ticket_issued_date'].apply(
    get_hour)

train_df['hearing_date_year'] = train_df['hearing_date'].apply(get_year)
train_df['hearing_date_month'] = train_df['hearing_date'].apply(get_month)
train_df['hearing_date_weekday'] = train_df['hearing_date'].apply(get_weekday)
train_df['hearing_date_day'] = train_df['hearing_date'].apply(get_day)
train_df['hearing_date_hour'] = train_df['hearing_date'].apply(get_hour)


train_df['ticket_issued_date_year'] = train_df['ticket_issued_date'].apply(
    get_year)
train_df['ticket_issued_date_month'] = train_df['ticket_issued_date'].apply(
    get_month)
train_df['ticket_issued_date_weekday'] = train_df['ticket_issued_date'].apply(
    get_weekday)
train_df['ticket_issued_date_day'] = train_df['ticket_issued_date'].apply(
    get_day)
train_df['ticket_issued_date_hour'] = train_df['ticket_issued_date'].apply(
    get_hour)

train_df['hearing_date_year'] = train_df['hearing_date'].apply(get_year)
train_df['hearing_date_month'] = train_df['hearing_date'].apply(get_month)
train_df['hearing_date_weekday'] = train_df['hearing_date'].apply(get_weekday)
train_df['hearing_date_day'] = train_df['hearing_date'].apply(get_day)
train_df['hearing_date_hour'] = train_df['hearing_date'].apply(get_hour)


train_df['ticket_issued_date_year'] = train_df['ticket_issued_date'].apply(
    get_year)
train_df['ticket_issued_date_month'] = train_df['ticket_issued_date'].apply(
    get_month)
train_df['ticket_issued_date_weekday'] = train_df['ticket_issued_date'].apply(
    get_weekday)
train_df['ticket_issued_date_day'] = train_df['ticket_issued_date'].apply(
    get_day)
train_df['ticket_issued_date_hour'] = train_df['ticket_issued_date'].apply(
    get_hour)

train_df['hearing_date_year'] = train_df['hearing_date'].apply(get_year)
train_df['hearing_date_month'] = train_df['hearing_date'].apply(get_month)
train_df['hearing_date_weekday'] = train_df['hearing_date'].apply(get_weekday)
train_df['hearing_date_day'] = train_df['hearing_date'].apply(get_day)
train_df['hearing_date_hour'] = train_df['hearing_date'].apply(get_hour)


test_df['ticket_issued_date_year'] = test_df['ticket_issued_date'].apply(
    get_year)
test_df['ticket_issued_date_month'] = test_df['ticket_issued_date'].apply(
    get_month)
test_df['ticket_issued_date_weekday'] = test_df['ticket_issued_date'].apply(
    get_weekday)
test_df['ticket_issued_date_day'] = test_df['ticket_issued_date'].apply(
    get_day)
test_df['ticket_issued_date_hour'] = test_df['ticket_issued_date'].apply(
    get_hour)

test_df['hearing_date_year'] = test_df['hearing_date'].apply(get_year)
test_df['hearing_date_month'] = test_df['hearing_date'].apply(get_month)
test_df['hearing_date_weekday'] = test_df['hearing_date'].apply(get_weekday)
test_df['hearing_date_day'] = test_df['hearing_date'].apply(get_day)
test_df['hearing_date_hour'] = test_df['hearing_date'].apply(get_hour)


# %% drop original datetime columns and save

train_df = train_df.drop(['ticket_issued_date', 'hearing_date'], axis=1)
train_df.to_csv('engineered_datasets/train_df_datetime.csv')

test_df = test_df.drop(['ticket_issued_date', 'hearing_date'], axis=1)
test_df.to_csv('engineered_datasets/test_df_datetime.csv')
