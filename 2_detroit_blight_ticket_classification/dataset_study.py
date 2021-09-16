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


#%% Preamble

import pandas as pd
# Make the output look better
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None  # default='warn' # ignores warning about dropping columns inplace
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns

import os
os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\2_detroit_blight_ticket_classification')


# !!!
#%% load datetime-processed training set


# get training set column names
train_cols_new = pd.read_csv('engineered_datasets/train_df_datetime.csv', nrows=0).columns
train_cols_new = train_cols_new.drop('Unnamed: 0').to_list()

# import datetime-processed training dataset
train_df_raw = pd.read_csv('engineered_datasets/train_df_datetime.csv', low_memory = True, usecols = train_cols_new, dtype = {'zip_code': str, 'non_us_str_code': str, 'grafitti_status': str})


# train_df_raw.shape
# (250306, 42)

train_df = train_df_raw.copy()

# import datetime-processed test dataset
test_df = pd.read_csv('engineered_datasets/test_df_datetime.csv', dtype = {'zip_code': str, 'non_us_str_code': str, 'grafitti_status': str, 'violation_zip_code': str, 'mailing_address_str_number': str})
test_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
# (61001, 35)



#%% general observations

train_df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 250306 entries, 0 to 250305
# Data columns (total 34 columns):
#  #   Column                      Non-Null Count   Dtype         
# ---  ------                      --------------   -----         
#  0   ticket_id                   250306 non-null  int64         
#  1   agency_name                 250306 non-null  object        
#  2   inspector_name              250306 non-null  object        
#  3   violator_name               250272 non-null  object        
#  4   violation_street_number     250306 non-null  float64       
#  5   violation_street_name       250306 non-null  object        
#  6   violation_zip_code          0 non-null       float64       
#  7   mailing_address_str_number  246704 non-null  float64       
#  8   mailing_address_str_name    250302 non-null  object        
#  9   city                        250306 non-null  object        
#  10  state                       250213 non-null  object        
#  11  zip_code                    250305 non-null  object        
#  12  non_us_str_code             3 non-null       object        
#  13  country                     250306 non-null  object        
#  14  ticket_issued_date          250306 non-null  datetime64[ns]
#  15  hearing_date                237815 non-null  datetime64[ns]
#  16  violation_code              250306 non-null  object        
#  17  violation_description       250306 non-null  object        
#  18  disposition                 250306 non-null  object        
#  19  fine_amount                 250305 non-null  float64       
#  20  admin_fee                   250306 non-null  float64       
#  21  state_fee                   250306 non-null  float64       
#  22  late_fee                    250306 non-null  float64       
#  23  discount_amount             250306 non-null  float64       
#  24  clean_up_cost               250306 non-null  float64       
#  25  judgment_amount             250306 non-null  float64       
#  26  payment_amount              250306 non-null  float64       
#  27  balance_due                 250306 non-null  float64       
#  28  payment_date                41113 non-null   object        
#  29  payment_status              250306 non-null  object        
#  30  collection_status           36897 non-null   object        
#  31  grafitti_status             1 non-null       object        
#  32  compliance_detail           250306 non-null  object        
#  33  compliance                  159880 non-null  float64       
# dtypes: datetime64[ns](2), float64(13), int64(1), object(18)
# memory usage: 64.9+ MB


train_df['compliance'].isnull().sum()
# compliance                     90426
# dtype: int64
"""
the target label has missing values; drop them
"""
train_df = train_df_raw[~np.isnan(train_df_raw['compliance'])] 


train_df.shape
# (159880, 42)
test_df.shape
# (61001, 35)

train_df.isnull().sum()
# ticket_id                          0
# agency_name                        0
# inspector_name                     0
# violator_name                     26
# violation_street_number            0
# violation_street_name              0
# violation_zip_code            159880
# mailing_address_str_number      2558
# mailing_address_str_name           3
# city                               0
# state                             84
# zip_code                           1
# non_us_str_code               159877
# country                            0
# violation_code                     0
# violation_description              0
# disposition                        0
# fine_amount                        0
# admin_fee                          0
# state_fee                          0
# late_fee                           0
# discount_amount                    0
# clean_up_cost                      0
# judgment_amount                    0
# payment_amount                     0
# balance_due                        0
# payment_date                  120269
# payment_status                     0
# collection_status             122983
# grafitti_status               159880
# compliance_detail                  0
# compliance                         0
# ticket_issued_date_year            0
# ticket_issued_date_month           0
# ticket_issued_date_weekday         0
# ticket_issued_date_day             0
# ticket_issued_date_hour            0
# hearing_date_year                227
# hearing_date_month               227
# hearing_date_weekday             227
# hearing_date_day                 227
# hearing_date_hour                227
# dtype: int64

test_df.isnull().sum()
# ticket_id                         0
# agency_name                       0
# inspector_name                    0
# violator_name                    28
# violation_street_number           0
# violation_street_name             0
# violation_zip_code            36977
# mailing_address_str_number     1014
# mailing_address_str_name          3
# city                              1
# state                           331
# zip_code                          3
# non_us_str_code               61001
# country                           0
# violation_code                    0
# violation_description             0
# disposition                       0
# fine_amount                       0
# admin_fee                         0
# state_fee                         0
# late_fee                          0
# discount_amount                   0
# clean_up_cost                     0
# judgment_amount                   0
# grafitti_status               58780
# ticket_issued_date_year           0
# ticket_issued_date_month          0
# ticket_issued_date_weekday        0
# ticket_issued_date_day            0
# ticket_issued_date_hour           0
# hearing_date_year              2197
# hearing_date_month             2197
# hearing_date_weekday           2197
# hearing_date_day               2197
# hearing_date_hour              2197
# dtype: int64

train_df.nunique()
# ticket_id                     159880
# agency_name                        5
# inspector_name                   159
# violator_name                  84656
# violation_street_number        18096
# violation_street_name           1716
# violation_zip_code                 0
# mailing_address_str_number     14090
# mailing_address_str_name       28440
# city                            4093
# state                             59
# zip_code                        4622
# non_us_str_code                    2
# country                            5
# violation_code                   189
# violation_description            207
# disposition                        4
# fine_amount                       40
# admin_fee                          1
# state_fee                          1
# late_fee                          37
# discount_amount                   13
# clean_up_cost                      1
# judgment_amount                   57
# payment_amount                   522
# balance_due                      606
# payment_date                    2307
# payment_status                     3
# collection_status                  1
# grafitti_status                    0
# compliance_detail                  8
# compliance                         2
# ticket_issued_date_year            9
# ticket_issued_date_month          12
# ticket_issued_date_weekday         7
# ticket_issued_date_day            31
# ticket_issued_date_hour           24
# hearing_date_year                 10
# hearing_date_month                12
# hearing_date_weekday               5
# hearing_date_day                  31
# hearing_date_hour                  5
# dtype: int64


test_df.nunique()
# ticket_id                     61001
# agency_name                       3
# inspector_name                  116
# violator_name                 38515
# violation_street_number       13999
# violation_street_name          1477
# violation_zip_code               58
# mailing_address_str_number    13192
# mailing_address_str_name      16851
# city                           3266
# state                            58
# zip_code                       2900
# non_us_str_code                   0
# country                           1
# violation_code                  151
# violation_description           163
# disposition                       8
# fine_amount                      53
# admin_fee                         1
# state_fee                         1
# late_fee                         44
# discount_amount                  14
# clean_up_cost                   298
# judgment_amount                 503
# grafitti_status                   1
# ticket_issued_date_year           5
# ticket_issued_date_month         12
# ticket_issued_date_weekday        7
# ticket_issued_date_day           31
# ticket_issued_date_hour          24
# hearing_date_year                 6
# hearing_date_month               12
# hearing_date_weekday              7
# hearing_date_day                 31
# hearing_date_hour                 4
# dtype: int64


"""
Observations
# ticket_id should not be taken as a feature to avoid data leakage, as it may contain information that has to do with the target
# few agency names; use one-hot encoding, i.e. get_dummies
# many inspectors

# many violators and their addresses
# violation_zip_code is mostly nan; drop
# graffiti_status is all nan in traning set, but not so for test set; drop
# payment_date: many nan, drop

['payment_amount','payment_date','payment_status','balance_due','collection_status','compliance_detail']: only in training set; drop

"""

# columns to drop
# 'ticket_id', 'violation_zip_code', 'graffiti_status'
# ['payment_amount','payment_date','payment_status','balance_due','collection_status','compliance_detail']

train_df.drop( ['payment_amount','payment_date','payment_status','balance_due','collection_status','compliance_detail'], axis = 1, inplace = True)

train_df.drop( ['ticket_id', 'violation_zip_code', 'grafitti_status'], axis = 1, inplace = True)
test_df.drop( ['ticket_id', 'violation_zip_code', 'grafitti_status'], axis = 1, inplace = True)

#%% target label

train_target = train_df[['compliance']].groupby(['compliance']).size()

fig = plt.figure('target', dpi = 150)
sns.barplot(x = train_target.index, y = train_target.values, color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Compliance')
ax.set_ylabel(None)
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig('plots/train_compliance.png', dpi = 150)

"""# !!! class imbalance"""


#%% ticket issue and hearing dates

"""
year
"""
train_df_year_count1 = train_df[['ticket_issued_date_year']].groupby(['ticket_issued_date_year']).size()
train_df_year_count1
# Out[43]: 
# ticket_issued_date_year
# 1988        1
# 2004       15
# 2005    25829
# 2006    28272
# 2007    27589
# 2008    29282
# 2009    22343
# 2010    15950
# 2011    10599
# dtype: int64

train_df = train_df[train_df['ticket_issued_date_year'] > 2004]
"""
drop year 1988, 2004
"""
train_df_year_count1 = train_df[['ticket_issued_date_year']].groupby(['ticket_issued_date_year']).size()

test_df_year_count1 = test_df[['ticket_issued_date_year']].groupby(['ticket_issued_date_year']).size()
test_df_year_count1
# ticket_issued_date_year
# 2012     8224
# 2013     7417
# 2014    11141
# 2015    17393
# 2016    16826
# dtype: int64


fig = plt.figure(dpi = 150)
sns.barplot(x = train_df_year_count1.index, y = train_df_year_count1.values, color = 'skyblue')
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
# plt.savefig('plots/1_datetime/ticket_year_training.png', dpi = 150)


fig = plt.figure(dpi = 150)
sns.barplot(x = test_df_year_count1.index, y = test_df_year_count1.values, color = 'tomato')
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
# plt.savefig('plots/1_datetime/ticket_year_test.png', dpi = 150)


train_df_year_count2 = train_df[['hearing_date_year']].groupby(['hearing_date_year']).size()
train_df_year_count2
# hearing_date_year
# 2005.0    20227
# 2006.0    30469
# 2007.0    25680
# 2008.0    28392
# 2009.0    23268
# 2010.0    18879
# 2011.0    11934
# 2012.0      784
# 2013.0        2
# 2016.0        2
# dtype: int64

test_df_year_count2 = test_df[['hearing_date_year']].groupby(['hearing_date_year']).size()
test_df_year_count2
# hearing_date_year
# 2012.0     7129
# 2013.0     7892
# 2014.0    10277
# 2015.0    16936
# 2016.0    15547
# 2017.0     1023
# dtype: int64


fig = plt.figure(dpi = 150)
sns.barplot(x = train_df_year_count2.index.astype(int), y = train_df_year_count2.values, color = 'skyblue')
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
# plt.savefig('plots/1_datetime/hearing_year_training.png', dpi = 150)


fig = plt.figure(dpi = 150)
sns.barplot(x = test_df_year_count2.index.astype(int), y = test_df_year_count2.values, color = 'tomato')
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
# plt.savefig('plots/1_datetime/hearing_year_test.png', dpi = 150)


"""
test set events happened after training set events
"""

"""
auto-correlations
"""


train_df_ticket_month_num = train_df[['ticket_month_num', 'compliance']].groupby(['ticket_month_num']).agg(np.nanmean).reset_index()


# from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, ax = plt.subplots(dpi = 150)
plot_acf(train_df_ticket_month_num['compliance'], color = 'skyblue', ax=ax)
ax.set_xlabel('Lag /month')
# ax.set_ylabel(None)
# ax.set_yticks([])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# plt.ylim(-0.25,)
# plt.savefig('plots/1_datetime/ticket_training_acf.png', dpi = 150)

fig, ax = plt.subplots(dpi = 150)
plot_pacf(train_df_ticket_month_num['compliance'], color = 'skyblue', ax=ax)
ax.set_xlabel('Lag /month')
# ax.set_ylabel(None)
# ax.set_yticks([])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# plt.ylim(-0.25,)
# plt.savefig('plots/1_datetime/ticket_training_pacf.png', dpi = 150)

"""
the [mean] compliance seems to be AR(1), but we are not looking at the mean compliance, so never mind lag features
"""

"""
drop year from features; will use tree-based models
"""

"""
month
"""


train_df_month_count1 = train_df[['ticket_issued_date_month']].groupby(['ticket_issued_date_month']).size()
# 1     12265
# 2     13995
# 3     15147
# 4     12639
# 5     13564
# 6     14298
# 7     13358
# 8     14943
# 9     15716
# 10    15463
# 11    10321
# 12     8155
# dtype: int64
""" very close"""


test_df_month_count1 = test_df[['ticket_issued_date_month']].groupby(['ticket_issued_date_month']).size()
test_df_month_count1
""" very close"""


fig = plt.figure(dpi = 150)
sns.barplot(x = train_df_month_count1.index, y = train_df_month_count1.values, color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Month')
ax.set_ylabel(None)
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(6000,)
# plt.savefig('plots/1_datetime/ticket_month_training.png', dpi = 150)


train_df_month_count2 = train_df[['hearing_date_month']].groupby(['hearing_date_month']).size()
train_df_month_count2
# hearing_date_month
# 1.0      9089
# 2.0     10976
# 3.0     14885
# 4.0     18840
# 5.0     13886
# 6.0     12515
# 7.0     12619
# 8.0     13960
# 9.0     11620
# 10.0    14379
# 11.0    11698
# 12.0    15170
# dtype: int64

test_df_year_count2 = test_df[['hearing_date_year']].groupby(['hearing_date_year']).size()
test_df_year_count2
# hearing_date_year
# 2012.0     7129
# 2013.0     7892
# 2014.0    10277
# 2015.0    16936
# 2016.0    15547
# 2017.0     1023
# dtype: int64


fig = plt.figure(dpi = 150)
sns.barplot(x = train_df_month_count2.index.astype(int), y = train_df_month_count2.values, color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Month')
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
plt.ylim(7500,)
# plt.savefig('plots/1_datetime/hearing_month_training.png', dpi = 150)

"""
mean by month
"""

train_df_monthly_comp1 = train_df[['ticket_issued_date_month', 'compliance']].groupby(['ticket_issued_date_month']).agg(np.nanmean).reset_index()



fig = plt.figure(dpi = 150)
sns.barplot(x = train_df_monthly_comp1['ticket_issued_date_month'], y = train_df_monthly_comp1['compliance'], color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Month')
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
plt.ylim(0.04,)
# plt.savefig('plots/1_datetime/ticket_month_mean_compl_training.png', dpi = 150)


train_df_monthly_comp2 = train_df[['hearing_date_month', 'compliance']].groupby(['hearing_date_month']).agg(np.nanmean).reset_index()



fig = plt.figure(dpi = 150)
sns.barplot(x = train_df_monthly_comp2['hearing_date_month'].astype(int), y = train_df_monthly_comp2['compliance'], color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Month')
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
# plt.ylim(0.04,)
# plt.savefig('plots/1_datetime/hearing_month_mean_compl_training.png', dpi = 150)


"""
mean encoding for ticket month
freq/mean for hearing month
"""

"""
day
"""


train_df_day_count1 = train_df[['ticket_issued_date_day']].groupby(['ticket_issued_date_day']).size()
""" very close"""


fig = plt.figure(dpi = 150)
sns.barplot(x = train_df_day_count1.index, y = train_df_day_count1.values, color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Day')
ax.set_ylabel(None)
ax.set_xticks([0,14,29])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(2000,)
# plt.savefig('plots/1_datetime/ticket_day_training.png', dpi = 150)


train_df_day_count2 = train_df[['hearing_date_day']].groupby(['hearing_date_day']).size()


fig = plt.figure(dpi = 150)
sns.barplot(x = train_df_day_count2.index.astype(int), y = train_df_day_count2.values, color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Day')
ax.set_xticks([0,14,29])
ax.set_ylabel(None)
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(2000,)
# plt.savefig('plots/1_datetime/hearing_day_training.png', dpi = 150)

"""
mean by day
"""

train_df_day_comp1 = train_df[['ticket_issued_date_day', 'compliance']].groupby(['ticket_issued_date_day']).agg(np.nanmean).reset_index()



fig = plt.figure(dpi = 150)
sns.barplot(x = train_df_day_comp1['ticket_issued_date_day'], y = train_df_day_comp1['compliance'], color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Day')
ax.set_xticks([0,14,29])
ax.set_ylabel(None)
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(0.06,)
# plt.savefig('plots/1_datetime/ticket_day_mean_compl_training.png', dpi = 150)


train_df_day_comp2 = train_df[['hearing_date_day', 'compliance']].groupby(['hearing_date_day']).agg(np.nanmean).reset_index()



fig = plt.figure(dpi = 150)
sns.barplot(x = train_df_day_comp2['hearing_date_day'].astype(int), y = train_df_day_comp2['compliance'], color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Day')
ax.set_xticks([0,14,29])
ax.set_ylabel(None)
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# plt.ylim(0.04,)
# plt.savefig('plots/1_datetime/hearing_day_mean_compl_training.png', dpi = 150)


"""
mean encoding for ticket day
freq for hearing day
"""


"""
weekday
"""


train_df_weekday_count1 = train_df[['ticket_issued_date_weekday']].groupby(['ticket_issued_date_weekday']).size().reset_index()
train_df_weekday_count1['ticket_issued_date_weekday'] =train_df_weekday_count1['ticket_issued_date_weekday'].replace(dict(zip(range(0, 7), ['Mon', 'Tue', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])))
train_df_weekday_count1.set_index('ticket_issued_date_weekday', inplace = True)

fig = plt.figure(dpi = 150)
sns.barplot(x = train_df_weekday_count1.index, y = train_df_weekday_count1.values.squeeze(), color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Weekday')
ax.set_ylabel(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# plt.ylim(2000,)
# plt.savefig('plots/1_datetime/ticket_weekday_training.png', dpi = 150)


train_df_weekday_count2 = train_df[['hearing_date_weekday']].groupby(['hearing_date_weekday']).size().reset_index()
train_df_weekday_count2['hearing_date_weekday'] =train_df_weekday_count2['hearing_date_weekday'].replace(dict(zip(range(0, 7), ['Mon', 'Tue', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])))
train_df_weekday_count2.set_index('hearing_date_weekday', inplace = True)


fig = plt.figure(dpi = 150)
sns.barplot(x = train_df_weekday_count2.index, y = train_df_weekday_count2.values.squeeze(), color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Weekday')
ax.set_ylabel(None)
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(2000,)
# plt.savefig('plots/1_datetime/hearing_weekday_training.png', dpi = 150)

"""
mean by weekday
"""

train_df_weekday_comp1 = train_df[['ticket_issued_date_weekday', 'compliance']].groupby(['ticket_issued_date_weekday']).agg(np.nanmean).reset_index()
train_df_weekday_comp1['ticket_issued_date_weekday'] =train_df_weekday_comp1['ticket_issued_date_weekday'].replace(dict(zip(range(0, 7), ['Mon', 'Tue', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])))
train_df_weekday_comp1.set_index('ticket_issued_date_weekday', inplace = True)


fig = plt.figure(dpi = 150)
sns.barplot(x = train_df_weekday_comp1.index, y = train_df_weekday_comp1.values.squeeze(), color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Weekday')
ax.set_ylabel(None)
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(0.06,)
# plt.savefig('plots/1_datetime/ticket_weekday_mean_compl_training.png', dpi = 150)



train_df_weekday_comp2 = train_df[['hearing_date_weekday', 'compliance']].groupby(['hearing_date_weekday']).agg(np.nanmean).reset_index()
train_df_weekday_comp2['hearing_date_weekday'] =train_df_weekday_comp2['hearing_date_weekday'].replace(dict(zip(range(0, 7), ['Mon', 'Tue', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])))
train_df_weekday_comp2.set_index('hearing_date_weekday', inplace = True)



fig = plt.figure(dpi = 150)
sns.barplot(x = train_df_weekday_comp2.index, y = train_df_weekday_comp2.values.squeeze(), color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Weekday')
# ax.set_xticks([0,14,29])
ax.set_ylabel(None)
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# plt.ylim(0.04,)
# plt.savefig('plots/1_datetime/hearing_weekday_mean_compl_training.png', dpi = 150)


"""
frequency encoding for ticket weekday
freq for hearing weekday
"""

"""
hour
"""


train_df_hour_count1 = train_df[['ticket_issued_date_hour']].groupby(['ticket_issued_date_hour']).size()
""" very close"""


fig = plt.figure(dpi = 150)
sns.barplot(x = train_df_hour_count1.index, y = train_df_hour_count1.values, color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Hour')
ax.set_ylabel(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# plt.savefig('plots/1_datetime/ticket_hour_training.png', dpi = 150)


train_df_hour_count2 = train_df[['hearing_date_hour']].groupby(['hearing_date_hour']).size()


fig = plt.figure(dpi = 150)
sns.barplot(x = train_df_hour_count2.index.astype(int), y = train_df_hour_count2.values, color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Hour')
ax.set_ylabel(None)
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(2000,)
# plt.savefig('plots/1_datetime/hearing_hour_training.png', dpi = 150)

"""
mean by hour
"""

train_df_hour_comp1 = train_df[['ticket_issued_date_hour', 'compliance']].groupby(['ticket_issued_date_hour']).agg(np.nanmean).reset_index()



fig = plt.figure(dpi = 150)
sns.barplot(x = train_df_hour_comp1['ticket_issued_date_hour'], y = train_df_hour_comp1['compliance'], color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Hour')
ax.set_ylabel(None)
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.ylim(0.06,)
# plt.savefig('plots/1_datetime/ticket_hour_mean_compl_training.png', dpi = 150)


train_df_hour_comp2 = train_df[['hearing_date_hour', 'compliance']].groupby(['hearing_date_hour']).agg(np.nanmean).reset_index()



fig = plt.figure(dpi = 150)
sns.barplot(x = train_df_hour_comp2['hearing_date_hour'].astype(int), y = train_df_hour_comp2['compliance'], color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Hour')
ax.set_ylabel(None)
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# plt.ylim(0.04,)
# plt.savefig('plots/1_datetime/hearing_hour_mean_compl_training.png', dpi = 150)


"""
mean encoding for ticket hour
freq for hearing hour
"""



#%% mailing address

address_cols = [
'mailing_address_str_number',
       'mailing_address_str_name', 'city', 'state', 'zip_code',
       'non_us_str_code', 'country'
]


train_df['country'].unique()
# array(['USA', 'Cana', 'Aust', 'Egyp', 'Germ'], dtype=object)

len(train_df[train_df['country'] != 'USA'])
# 11

train_df[train_df['country'] != 'USA'][address_cols]
#         mailing_address_str_number      mailing_address_str_name  \
# 160652                         2.0                   ANTIQUE DR.   
# 177864                       238.0            ROBINA TOWN CENTER   
# 211755                     25359.0                   16TH AVENUE   
# 216567                         1.0                        TAIMOR   
# 216568                         1.0                     TAIMOR ST   
# 216927                        47.0                 ELMONT  DRIVE   
# 222230                         NaN                   P O BOX 717   
# 226259                         NaN         ROSA-REINGLASS-STEIG2   
# 226609                        65.0  WYNFORD HTS. CRES, STE. 1905   
# 226610                        65.0  WYNFORD HTS. CRES, STE. 1905   
# 236075                       111.0                     LIVERPOOL   

#                                            city state    zip_code  \
# 160652                            RICHMOND HILL    ON      L4E3V8   
# 177864                               QUEENSLAND    QL        4226   
# 211755                               ALDERGROVE    BC      V4W2R7   
# 216567  ST. FATIMA SQ. HELIOPOLIS, CAIRO, EGYPT   NaN       11361   
# 216568  ST. FATIMA SQ. HELIOPOLIS, CAIRO, EGYPT   NaN       11361   
# 216927                 CALGARY, ALBERTA, CANADA   NaN     T3H-4X8   
# 222230                                SUN RIDGE    ON      POAIZO   
# 226259                                   BERLIN    BL       13585   
# 226609                                  TORONTO    ON  M3C1L-7000   
# 226610                                  TORONTO    ON  M3C1L-7000   
# 236075                      ST ROSEBAY AUSTRALI   NaN     NSW2029   

#         non_us_str_code country  
# 160652              NaN    Cana  
# 177864      , Australia    Aust  
# 211755              NaN    Cana  
# 216567              NaN    Egyp  
# 216568              NaN    Egyp  
# 216927              NaN    Cana  
# 222230              NaN    Cana  
# 226259              NaN    Germ  
# 226609  ONTARIO, Canada    Cana  
# 226610  ONTARIO, Canada    Cana  
# 236075              NaN    Aust  
"""# really not in the US"""

train_df[train_df['country'] == 'USA']['state'].unique()
# array(['IL', 'MI', 'CA', 'NY', 'MN', 'NV', 'PA', 'LA', 'MD', 'FL', 'ME',
#        'KY', 'TX', 'AZ', 'TN', 'OH', 'GA', 'IN', 'MS', 'NJ', 'WA', 'WI',
#        'UT', 'VA', 'SC', 'MO', 'AL', 'DC', 'CT', 'AR', 'OK', 'MA', 'CO',
#        'UK', 'NC', 'AK', 'RI', 'NM', 'VT', 'NB', 'MT', 'IA', 'ID', 'OR',
#        'DE', 'PR', 'NH', 'VI', 'KS', 'SD', 'QC', 'ON', 'HI', nan, 'WY',
#        'WV', 'ND', 'BC', 'QL'], dtype=object)
"""has non-US states (e.g. QC, BC); miscategorised as USA """

train_df['zip_code'].head(20)


"""zip_code has non-5-digit instances"""

"""look at zipcodes where 'state' is really one of the 50 (+DC) states """
US_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
              "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
              "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
              "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
              "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]


train_df['zip_code'] = train_df['zip_code'].fillna('nfd')

def country_zip_func0(row) : 
    if row['state'] in US_states:
        row['zip_len'] = len(row['zip_code'])
    else:
            row['zip_len'] = -1
    return row

train_df = train_df.apply(country_zip_func0, axis=1)


train_df['zip_len'][train_df['zip_len'] != 5]
# Out[164]: 
# 455       6
# 654       4
# 764       3
# 877       4
# 1328      4
#          ..
# 249505   -1
# 249784    6
# 250034   -1
# 250070   -1
# 250136   -1
# Name: zip_len, Length: 1129, dtype: int64

train_df['zip_len'].unique()
# array([ 5,  6,  4,  3,  9,  1, -1,  2, 10,  7], dtype=int64)

train_df[train_df['zip_len'] == 9]['zip_code'].head()
# Out[118]: 
# 2081     480210433
# 11199    190403430
# 13261    482355555
# 24629    770812212
# 33417    481743208
# Name: zip_code, dtype: object


train_df[train_df['zip_len'] == 10]['zip_code'].head()
# Out[140]: 
# 42394     92637-2854
# 58955     60654-6939
# 108365    48021-1319
# 109116    48228-1574
# 110833    60008-4227
# Name: zip_code, dtype: object

train_df[train_df['zip_len'] == 7][address_cols]
# Out[168]: 
#         mailing_address_str_number mailing_address_str_name    city state  \
# 57355                          1.0           CITY BLVD WEST  ORANGE    CA   
# 114955                        11.0              COUNTRY RD.  ORANGE    CA   

#        zip_code non_us_str_code country  
# 57355   9288689             NaN     USA  
# 114955  4892868             NaN     USA  

"""zip codes should be 92868"""

train_df[train_df['zip_len'] == 6][address_cols].head()
# 113 of them
"""the given zip codes do not match the address, but there aren't many of them, nvm"""



""" instances in any of the US states, with 9 or 10 digit for zip_code, make sense"""




train_df_state = train_df[['state']].groupby(['state']).size()
train_df_state
# mostly MI, obviously




# !!!
"""
# many mailing addresses, few (11) outside the US
# Many mis-categorised countries into 'USA': country just USA or not
# zip_code needs processing: just take first 5 digits using regex, including those wrong zip codes (6-7 digits) which are not many
frequency encode state
"""



#%% violation_code and violation_description


"""
# violation_description is redundant
"""




#%% fine and fees



"""
# admin_fee and state_fee are all the same
# clean_up_cost are all zero, but test data has a lot of non-zeros
# judgment_amount is a simple sum of fees

"""

# !!!
#%% dropping outliers

train_df = train_df[~np.isnan(train_df['compliance'])] 
train_df.drop( ['payment_amount','payment_date','payment_status','balance_due','collection_status','compliance_detail'], axis = 1, inplace = True)
train_df.drop( ['ticket_id', 'violation_zip_code', 'grafitti_status'], axis = 1, inplace = True)
train_df = train_df[train_df['ticket_issued_date_year'] > 2004]
train_df['zip_code'] = train_df['zip_code'].fillna('nfd')



test_df.drop( ['ticket_id', 'violation_zip_code', 'grafitti_status'], axis = 1, inplace = True)







