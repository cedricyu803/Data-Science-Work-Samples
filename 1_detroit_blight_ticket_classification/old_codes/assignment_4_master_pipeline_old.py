# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 20:00:00 2021

@author: Cedric Yu
"""

"""
Pipeline version. For non-pipeline version, see files in Week 4 folder.
"""

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
pd.options.mode.chained_assignment = None  # default='warn' # ignores warning about dropping columns inplace
import numpy as np
import matplotlib.pyplot as plt


#%% pre-processing training dataset train.csv

# addresses_df = pd.read_csv('addresses.csv')
# latlons_df = pd.read_csv('latlons.csv')

# import original training dataset
train_df_all = pd.read_csv('train.csv', encoding = 'ISO-8859-1' , low_memory=False)
train_df_all.reset_index(inplace=True)
train_df_all.drop('index', inplace=True, axis=1)
# train_df_all.shape # (250306, 34)
# column 11, i.e. 'zip_code' is taken as object dtype due to hyphens and English alphebets
# column 12, i.e. 'non_us_str_code', and 31, i.e. 'grafitti_status', are correctly taken as object dtype

train_df = train_df_all[~np.isnan(train_df_all['compliance'])] 
# drop instances with NaN compliance
train_df.reset_index(inplace=True)
train_df.drop('index', inplace=True,axis=1)
# drop columns that are only accessible when compliance is made
train_df.drop(['payment_amount','payment_date','payment_status','balance_due','collection_status','compliance_detail'], inplace=True,axis=1)

#%% train-validation split. separate train_df into features X and target y

from sklearn.model_selection import train_test_split

X = train_df.drop('compliance', axis=1)
y = train_df['compliance']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

#%% pre-processing with ColumnTransformer and custom estimators

# import pre-processing custom estimators and ColumnTransformer
import assignment_4_preprocessing as pp

#%% Pipeline

# define model
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(random_state=0)

# define pipeline
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('preprocessor', pp.preprocessor),
    ('classifier', model)
    ])

# pipe.fit(X_train, y_train)

# parameters for GridSearchCV
parameters = {'classifier__learning_rate': [0.01, 0.1, 0.5]
}

from sklearn.model_selection import GridSearchCV # For optimization
# pass them to GradSearchCV and fit with training data
grid = GridSearchCV(pipe, parameters, cv=5, scoring = 'roc_auc').fit(X_train, y_train)
# grid.best_params_
# {'classifier__learning_rate': 0.5}
# grid.best_score_
# 0.824706542844932
print('Training set score: ' + str(grid.score(X_train, y_train)))
print('Test set score: ' + str(grid.score(X_valid, y_valid)))
# Training set score: 0.8436171343313693
# Test set score: 0.823325217661963

best_params = grid.best_params_
best_pipe = grid.best_estimator_

#%% model scores

from sklearn.metrics import roc_curve, auc

y_valid_predict_proba = best_pipe.predict_proba(X_valid)

fpr_clf, tpr_clf, thresholds_clf  = roc_curve(y_valid, y_valid_predict_proba[:,1])
auc_clf = auc(fpr_clf, tpr_clf)

print('RCO-AUC scores: \n')
print('Gradient boost: {:.3f}\n'.format(auc_clf))
# RCO-AUC scores: 

# Gradient boost: 0.823

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_clf, tpr_clf, lw=3, label='Gradient boost ROC curve (area = {:0.2f})'.format(auc_clf))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()

#%% feature importances

clf_feature_importances = pd.Series(best_pipe.steps[1][1].feature_importances_).sort_values(ascending=False)

print(clf_feature_importances.head(10))

# 194    0.428650
# 190    0.216861
# 195    0.144675
# 1      0.084641
# 191    0.044162
# 0      0.025486
# 193    0.014879
# 192    0.006241
# 2      0.004713
# 144    0.003595
# dtype: float64


#%%

test_df_all = pd.read_csv('test.csv', encoding = 'ISO-8859-1')


pipe.predict(test_df_all)
# array([0., 0., 0., ..., 0., 0., 1.])

y_test_predict_proba = pd.Series(pipe.predict_proba(test_df_all)[:,1], index= test_df_all['ticket_id'], name='compliance')








