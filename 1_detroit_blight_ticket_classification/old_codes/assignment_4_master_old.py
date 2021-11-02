# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 17:00:00 2021

@author: Cedric Yu
"""

# import matplotlib.pyplot as plt
# import math
# import re

# Your AUC of 0.774448781327 was awarded a value of 1.0 out of 1.0 total grades

"""
# Note: when submitting on Coursera, use get_dummies instead of one-hot encoder

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
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt


import os
os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\2_detroit_blight_ticket_classification')


#%% pre-processing training dataset train.csv

# addresses_df = pd.read_csv('addresses.csv')
# latlons_df = pd.read_csv('latlons.csv')

# import original training dataset
train_df_all = pd.read_csv('train.csv', encoding = 'ISO-8859-1' , low_memory=False)
train_df_all.reset_index(inplace=True)
train_df_all.drop('index', inplace=True, axis=1)
# train_df_all.shape # (250306, 34)
# column 11, i.e. 'zip_code' is taken as object dtype due to hyphens
# column 12, i.e. 'non_us_str_code', and 31, i.e. 'grafitti_status', are correctly taken as object dtype


train_df = train_df_all[~np.isnan(train_df_all['compliance'])] 
# drop instances with NaN compliance
train_df.reset_index(inplace=True)
train_df.drop('index', inplace=True,axis=1)
# drop columns that are only accessible when compliance is made
train_df.drop(['payment_amount','payment_date','payment_status','balance_due','collection_status','compliance_detail'], inplace=True,axis=1)


#%% import pre-processing custom module
# See documentation inside the module

import assignment_4_preprocessing as pp
# assignment_4_preprocessing2 drops all categorical columns

#%% pick useful features and define target of the train-validation set

X = pp.drop_cols(train_df).drop('compliance', axis=1)
y = train_df['compliance']

#%% train-validation split

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y)

#%% apply pre-processing

OH_X_train = pp.train_preprocess(X_train)
OH_X_valid = pp.not_train_preprocess(X_valid)

# checked that all dtypes are int64 or float64
# OH_X_train.apply(dtype,axis=0)
# OH_X_valid.apply(dtype,axis=0)

#%% scaler

# OH_X_train_scaled = pp.train_scaler(OH_X_train)
# OH_X_valid_scaled = pp.non_train_scaler(OH_X_valid)


#%% model scores

from sklearn.metrics import roc_curve, auc

from sklearn.ensemble import GradientBoostingClassifier
clf_gb = GradientBoostingClassifier().fit(OH_X_train, y_train)
y_valid_predict_proba_clf_gb = clf_gb.predict_proba(OH_X_valid)

fpr_clf_gb, tpr_clf_gb, thresholds_clf_gb  = roc_curve(y_valid, y_valid_predict_proba_clf_gb[:,1])
auc_clf_gb = auc(fpr_clf_gb, tpr_clf_gb)

print('RCO-AUC scores: \n')
print('Gradient boost: {:.3f}\n'.format(auc_clf_gb))

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_clf_gb, tpr_clf_gb, lw=3, label='Gradient boost ROC curve (area = {:0.2f})'.format(auc_clf_gb))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()

#%% feature importances

clf_gb_feature_importances = pd.Series(clf_gb.feature_importances_, index=OH_X_train.columns).sort_values(ascending=False)

print(clf_gb_feature_importances.head(10))

# RCO-AUC scores: 

# Gradient boost: 0.815

# late_fee              0.484177
# 183                   0.226377
# discount_amount       0.157763
# 184                   0.050407
# ticket_issued_date    0.047565
# fine_amount           0.009194
# 139                   0.006009
# 185                   0.004048
# zip_code              0.004018
# 36                    0.001927
# dtype: float64


#%%

test_df_all = pd.read_csv('test.csv', encoding = 'ISO-8859-1')

X_pred = pp.drop_cols(test_df_all)
OH_X_pred = pp.not_train_preprocess(X_pred)

clf_gb.predict(OH_X_pred)
# array([0., 0., 0., ..., 0., 0., 1.])

y_test_predict_proba = pd.Series(clf_gb.predict_proba(OH_X_pred)[:,1], index= test_df_all['ticket_id'], name='compliance')


# import numpy as np
# bm = blight_model()
# res = '{:40s}'.format('Object Type:')
# res += ['Failed: type(bm) should Series\n','Passed\n'][type(bm)==pd.Series]
# res += '{:40s}'.format('Data Shape:')
# res += ['Failed: len(bm) should be 61001\n','Passed\n'][len(bm)==61001]
# res += '{:40s}'.format('Data Values Type:')
# res += ['Failed: bm.dtype should be float\n','Passed\n'][str(bm.dtype).count('float')>0]
# res += '{:40s}'.format('Data Values Infinity:')
# res += ['Failed: values should not be infinity\n','Passed\n'][not any(np.isinf(bm))]
# res += '{:40s}'.format('Data Values NaN:')
# res += ['Failed: values should not be NaN\n','Passed\n'][not any(np.isnan(bm))]
# res += '{:40s}'.format('Data Values in [0,1] Range:')
# res += ['Failed: all values should be in [0.,1.]\n','Passed\n'][all((bm<=1.) & (bm>=0.))]
# res += '{:40s}'.format('Data Values not all 0 or 1:')
# res += ['Failed: values should be scores not predicted labels\n','Passed\n'][not all((bm.isin({0,1,0.0,1.0})))]
# res += '{:40s}'.format('Index Type:')
# res += ['Failed: type(bm.index) should be Int64Index\n','Passed\n'][type(bm.index)==pd.Int64Index]
# res += '{:40s}'.format('Index Values:')
# if bm.index.shape==(61001,):
#     res +=['Failed: index values should match test.csv\n','Passed\n'
#           ][all(pd.read_csv('test.csv',usecols=[0],index_col=0
#                            ).sort_index().index.values==bm.sort_index().index.values)]
# else:
#     res+='Failed: bm.index length should be 61001'
# res += '{:40s}'.format('Can run model twice:')
# bm2 = None
# try:
#     bm2 = blight_model()
#     res += 'Passed\n'
# except:
#     res += ['Failed: second run of blight_model() threw an Exception']
# res += '{:40s}'.format('Can run model twice with same results:')
# if not bm2 is None:
#     res += ['Failed: second run of blight_model() produced different results (this might not be a problem)\n','Passed\n'][
#         all(bm.apply(lambda x:round(x,3))==bm2.apply(lambda x:round(x,3))) and all(bm.index==bm2.index)]    
# print(res)








