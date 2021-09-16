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
import seaborn as sns
import re
import gc

import os
os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\2_detroit_blight_ticket_classification')


#%% load pre-processed datasets for model training
# !!!

X_train_encoded4_scaled = pd.read_csv('engineered_datasets/X_train_encoded4_scaled.csv', index_col = [0])
X_valid_encoded4_scaled = pd.read_csv('engineered_datasets/X_valid_encoded4_scaled.csv', index_col = [0])

test_id = pd.read_csv('engineered_datasets/test_df_datetime.csv', usecols = [1])

X_test_encoded4_scaled = pd.read_csv('engineered_datasets/X_test_encoded4_scaled.csv', index_col = [0])

y_train = pd.read_csv('engineered_datasets/y_train.csv', index_col = [0]).squeeze()
y_valid = pd.read_csv('engineered_datasets/y_valid.csv', index_col = [0]).squeeze()

gc.collect()


#%% model scores

from sklearn.metrics import roc_curve, auc

#################################

from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy = 'most_frequent').fit(X_train_encoded4_scaled, y_train)

y_valid_predict_proba_dummy_clf = dummy_clf.predict_proba(X_valid_encoded4_scaled)

fpr_dummy_clf, tpr_dummy_clf, thresholds_dummy_clf  = roc_curve(y_valid, y_valid_predict_proba_dummy_clf[:,1])
auc_dummy_clf = auc(fpr_dummy_clf, tpr_dummy_clf)

print('ROC-AUC scores:')
print('Dummy classifier: {:.3f}\n'.format(auc_dummy_clf))
# ROC-AUC scores: 
# Dummy classifier: 0.500

plt.figure(figsize = (10, 10), dpi = 150)
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_dummy_clf, tpr_dummy_clf, lw=3, label='Dummy classifier ROC curve (area = {:0.2f})'.format(auc_dummy_clf))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
# plt.axes().set_aspect('equal')
plt.show()




#################################

from sklearn.neighbors import KNeighborsClassifier
k = 7
knnclf = KNeighborsClassifier(n_neighbors = k)
knnclf.fit(X_train_encoded4_scaled, y_train)


y_valid_predict_proba_knnclf = knnclf.predict_proba(X_valid_encoded4_scaled)

fpr_knnclf, tpr_knnclf, thresholds_knnclf  = roc_curve(y_valid, y_valid_predict_proba_knnclf[:,1])
auc_knnclf = auc(fpr_knnclf, tpr_knnclf)

print('ROC-AUC scores:')
print('KNN classifier: {:.3f}\n'.format(auc_knnclf))
# ROC-AUC scores:
# KNN classifier: 0.727

plt.figure(figsize = (6, 6), dpi = 150)
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_knnclf, tpr_knnclf, lw=3, label='KNN classifier (k={}) ROC curve (area = {:0.3f})'.format(k, auc_knnclf))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.tight_layout()
# plt.savefig('plots/KNN_roc', dpi = 150)

# np.save('ROC/fpr_knnclf', fpr_knnclf)
# np.save('ROC/tpr_knnclf', tpr_knnclf)
# np.save('ROC/auc_knnclf', auc_knnclf)


from sklearn.model_selection import GridSearchCV

knnclf2 = KNeighborsClassifier()
grid_values_knn = {'n_neighbors': [4, 5, 6, 7]}
# grid_values_knn = {'n_neighbors': [4, 5, 6, 7], 'p': [1, 2, 3]}

knnclf2_roc_auc = GridSearchCV(knnclf2, param_grid = grid_values_knn, scoring='roc_auc', cv = 3, verbose = 1)
knnclf2_roc_auc.fit(X_train_encoded4_scaled, y_train)
print('Grid best parameter (roc_auc): ', knnclf2_roc_auc.best_params_)
# Grid best parameter (roc_auc):  {'n_neighbors': 7}



#################################


from xgboost import XGBClassifier



# XGBR_model = XGBRegressor(eval_metric = "rmse", 
#                           learning_rate = 0.05, 
#                           max_depth = 8,
#                           n_estimators = 100,
#                           reg_lambda = 0.7,
#                           n_jobs = 6)


XGBC_model = XGBClassifier(scale_pos_weight = (127891.-9247.)/9247., 
                           eval_metric='auc', 
                           n_estimators = 900, 
                           max_depth = 7,
                           learning_rate = 0.01,
                           n_jobs = 8)
XGBC_model.fit(X_train_encoded4_scaled, y_train, eval_set = [(X_train_encoded4_scaled, y_train), (X_valid_encoded4_scaled, y_valid)], early_stopping_rounds = 40)


y_valid_predict_proba_XGBC_model = XGBC_model.predict_proba(X_valid_encoded4_scaled)

fpr_XGBC_model, tpr_XGBC_model, thresholds_XGBC_model  = roc_curve(y_valid, y_valid_predict_proba_XGBC_model[:,1])
auc_XGBC_model = auc(fpr_XGBC_model, tpr_XGBC_model)

print('ROC-AUC scores:')
print('XGBoost classifier: {:.3f}\n'.format(auc_XGBC_model))
# ROC-AUC scores:
# XGBoost classifier: 0.833


plt.figure(figsize = (6, 6), dpi = 150)
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_XGBC_model, tpr_XGBC_model, lw=3, label='XGBoost classifier ROC curve (area = {:0.3f})'.format(auc_XGBC_model))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.tight_layout()
# plt.savefig('plots/XGBC_model_roc', dpi = 150)

# np.save('ROC/fpr_XGBC_model', fpr_XGBC_model)
# np.save('ROC/tpr_XGBC_model', tpr_XGBC_model)
# np.save('ROC/auc_XGBC_model', auc_XGBC_model)



XGBR_feature_importances = pd.Series(XGBC_model.feature_importances_, index = X_train_encoded4_scaled.columns).sort_values(ascending = False)
# disposition_Responsible by Default                            0.590807
# late_fee                                                      0.140346
# disposition_Responsible by Admission                          0.078259
# discount_amount                                               0.044505
# fine_amount                                                   0.016901
# violation_code_freq_encoded                                   0.012649
# agency_name_Buildings, Safety Engineering & Env Department    0.012150
# hearing_date_month_mean_encoded                               0.011165
# agency_name_Department of Public Works                        0.010995
# country                                                       0.008593
# hearing_date_weekday_freq_encoded                             0.007548
# state_freq_encoded                                            0.007116
# zip_code                                                      0.006976
# ticket_issued_date_month_mean_encoded                         0.006904
# agency_name_Health Department                                 0.006801
# hearing_date_day_freq_encoded                                 0.006769
# ticket_issued_date_day_mean_encoded                           0.005781
# ticket_issued_date_hour_mean_encoded                          0.005722
# ticket_issued_date_weekday_freq_encoded                       0.005430
# agency_name_Detroit Police Department                         0.005387
# hearing_date_hour_freq_encoded                                0.005267
# disposition_Responsible by Determination                      0.003929
# disposition_Responsible (Fine Waived) by Deter                0.000000
# agency_name_Neighborhood City Halls                           0.000000
# dtype: float32


plt.figure(figsize = (12, 6), dpi = 150)
XGBR_feature_importances[0:10].plot.barh(color = 'skyblue')
plt.xlabel(None)
plt.ylabel(None)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
# plt.savefig('plots/XGBM_feature_imp', dpi = 150)

"""RandomizedSearchCV """
from sklearn.model_selection import RandomizedSearchCV

grid_values_XGBC = {'n_estimators': list(np.arange(100, 1100, 100)), 'learning_rate': [0.001, 0.01, 0.1], 'max_depth': [3, 4, 5, 6, 7]}

fit_params={"eval_metric" : "auc", 
            "eval_set" : [[X_valid_encoded4_scaled, y_valid]]}

XGBC_model2 = XGBClassifier(eval_metric="auc", n_jobs = 8)

XGBC_model_search = RandomizedSearchCV(XGBC_model2, param_distributions = grid_values_XGBC, scoring = 'roc_auc', cv = 3, verbose = 1, n_iter = 10)
XGBC_model_search.fit(X_train_encoded4_scaled, y_train, **fit_params)
print('Grid best parameters (roc_auc): ', XGBC_model_search.best_params_)
# Grid best parameters (roc_auc):  {'n_estimators': 900, 'max_depth': 7, 'learning_rate': 0.01}


#################################


from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(class_weight = 'balanced', 
                             n_estimators = 700, 
                             max_depth = 20, 
                             max_features= 'auto', 
                             min_samples_leaf = 1, 
                             min_samples_split = 16, 
                             random_state = 0, 
                             n_jobs = -1)
RFC.fit(X_train_encoded4_scaled, y_train)


# print(accuracy_score(y_train, RFC.predict(encoded_X_train_scaled)))
# print(accuracy_score(y_valid, RFC.predict(encoded_X_valid_scaled)))
# 0.9805389221556886
# 0.8251121076233184


y_valid_predict_proba_RFC = RFC.predict_proba(X_valid_encoded4_scaled)

fpr_RFC, tpr_RFC, thresholds_RFC  = roc_curve(y_valid, y_valid_predict_proba_RFC[:,1])
auc_RFC = auc(fpr_RFC, tpr_RFC)

print('ROC-AUC scores:')
print('RandomForestclassifier: {:.3f}\n'.format(auc_RFC))
# ROC-AUC scores:
# RandomForestclassifier: 0.836


plt.figure(figsize = (6, 6), dpi = 150)
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_RFC, tpr_RFC, lw=3, label='RandomForestclassifier ROC curve (area = {:0.3f})'.format(auc_RFC))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.tight_layout()
# plt.savefig('plots/RFC_roc', dpi = 150)

np.save('ROC/fpr_RFC', fpr_RFC)
np.save('ROC/tpr_RFC', tpr_RFC)
np.save('ROC/auc_RFC', auc_RFC)


RFC_feature_importances = pd.Series(RFC.feature_importances_, index = original_cols).sort_values(ascending = False)
RFC_feature_importances = RFC_feature_importances / RFC_feature_importances.max()
# late_fee                                                      1.000000
# disposition_Responsible by Default                            0.861825
# zip_code                                                      0.510369
# hearing_date_day_freq_encoded                                 0.415773
# ticket_issued_date_day_mean_encoded                           0.406555
# hearing_date_month_mean_encoded                               0.335010
# disposition_Responsible by Admission                          0.312385
# ticket_issued_date_month_mean_encoded                         0.289926
# disposition_Responsible by Determination                      0.273012
# ticket_issued_date_hour_mean_encoded                          0.244317
# violation_code_freq_encoded                                   0.202259
# ticket_issued_date_weekday_freq_encoded                       0.200476
# hearing_date_weekday_freq_encoded                             0.195399
# fine_amount                                                   0.187910
# hearing_date_hour_freq_encoded                                0.156705
# discount_amount                                               0.125300
# state_freq_encoded                                            0.098981
# agency_name_Buildings, Safety Engineering & Env Department    0.039194
# agency_name_Department of Public Works                        0.036030
# agency_name_Health Department                                 0.018464
# disposition_Responsible_Fine Waived_by_Deter                  0.015707
# agency_name_Detroit Police Department                         0.013724
# country                                                       0.003514
# agency_name_Neighborhood City Halls                           0.000007
# dtype: float64


plt.figure(figsize = (12, 6), dpi = 150)
RFC_feature_importances[0:10].plot.barh(color = 'skyblue')
plt.xlabel(None)
plt.ylabel(None)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
# plt.savefig('plots/RFC_feature_imp', dpi = 150)


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = np.arange(100, 1050, 50)
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [10, 20, 40, 60, 80, 100, None]
# Minimum number of samples required to split a node
min_samples_split = [2, 8, 16, 128, 1024, 4096]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 4, 128, 512, 1024, 4096]

# from sklearn.metrics import SCORERS

# Create the random grid
random_grid_RFC = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf}


RFC_model = RandomForestClassifier(random_state = 0, n_jobs = -1)
RFC_model_search = RandomizedSearchCV(RFC_model, param_distributions = random_grid_RFC, scoring = 'roc_auc', cv = 3, verbose = 1, n_iter = 6)
RFC_model_search.fit(X_train_encoded4_scaled, y_train)
print('Grid best parameter (auc): ', RFC_model_search.best_params_)



# from pickle import dump
# # save the model
# dump(RFC_best, open('RFC_best.pkl', 'wb'))



#################################

import lightgbm as lgb

# column names have unsupported characters
original_cols = X_train_encoded4_scaled.columns
X_train_encoded4_scaled.columns = np.arange(0,24)
X_valid_encoded4_scaled.columns = np.arange(0,24)
X_test_encoded4_scaled.columns = np.arange(0,24)

LGBMclf = lgb.LGBMClassifier(class_weight = 'balanced', 
                             learning_rate = 0.01, 
                            num_leaves = 800,
                            n_estimators = 500, 
                            num_iterations = 900, 
                            max_bin = 500, 
                            feature_fraction = 0.7, 
                            bagging_fraction = 0.7,
                            lambda_l2 = 0.5,
                            max_depth = 7,
                            silent = False
                            )

LGBMclf.fit(X_train_encoded4_scaled, y_train,
            eval_set=[(X_train_encoded4_scaled, y_train), (X_valid_encoded4_scaled, y_valid)], 
            eval_metric = 'auc',
            early_stopping_rounds = 50, 
            verbose = True)


y_test_pred = LGBMclf.predict(X_test_encoded4_scaled).astype(int)
y_pred = pd.Series(y_test_pred, index = test_id.values.squeeze())
y_pred.index.name = 'ticket_id'
y_pred.name = 'compliance'
y_pred.to_csv('y_pred_lgbm.csv')

test_target = y_pred.to_frame()[['compliance']].groupby(['compliance']).size()

fig = plt.figure('target', dpi = 150)
sns.barplot(x = test_target.index, y = test_target.values, color = 'tomato')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Compliance')
ax.set_ylabel(None)
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig('plots/test_compliance.png', dpi = 150)

y_valid_predict_proba_LGBMclf = LGBMclf.predict_proba(X_valid_encoded4_scaled)

fpr_LGBMclf, tpr_LGBMclf, thresholds_LGBMclf  = roc_curve(y_valid, y_valid_predict_proba_LGBMclf[:,1])
auc_LGBMclf = auc(fpr_LGBMclf, tpr_LGBMclf)

print('ROC-AUC scores:')
print('LGBM classifier: {:.3f}\n'.format(auc_LGBMclf))
# ROC-AUC scores:
# LGBM classifier: 0.836


plt.figure(figsize = (7, 7), dpi = 150)
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_LGBMclf, tpr_LGBMclf, lw=3, label='LGBM classifier ROC curve (area = {:0.3f})'.format(auc_LGBMclf))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.tight_layout()
# plt.savefig('plots/LGBM_roc', dpi = 150)

# np.save('ROC/fpr_LGBMclf', fpr_LGBMclf)
# np.save('ROC/tpr_LGBMclf', tpr_LGBMclf)
# np.save('ROC/auc_LGBMclf', auc_LGBMclf)


LGBMclf_feature_importances = pd.Series(LGBMclf.feature_importances_, index = original_cols).sort_values(ascending = False)
LGBMclf_feature_importances = LGBMclf_feature_importances / LGBMclf_feature_importances.max()

# zip_code                                                      1.000000
# ticket_issued_date_day_mean_encoded                           0.637136
# hearing_date_day_freq_encoded                                 0.617271
# ticket_issued_date_month_mean_encoded                         0.453423
# violation_code_freq_encoded                                   0.364653
# hearing_date_month_mean_encoded                               0.337897
# ticket_issued_date_hour_mean_encoded                          0.314183
# fine_amount                                                   0.274720
# hearing_date_weekday_freq_encoded                             0.267025
# ticket_issued_date_weekday_freq_encoded                       0.216644
# hearing_date_hour_freq_encoded                                0.210828
# state_freq_encoded                                            0.175391
# late_fee                                                      0.149799
# discount_amount                                               0.062819
# disposition_Responsible by Admission                          0.060850
# agency_name_Buildings, Safety Engineering & Env Department    0.060313
# disposition_Responsible by Default                            0.058702
# agency_name_Department of Public Works                        0.052170
# disposition_Responsible by Determination                      0.040000
# agency_name_Health Department                                 0.034362
# agency_name_Detroit Police Department                         0.016734
# disposition_Responsible_Fine Waived_by_Deter                  0.007785
# country                                                       0.007069
# agency_name_Neighborhood City Halls                           0.000000
# dtype: float64



plt.figure(figsize = (12, 6), dpi = 150)
LGBMclf_feature_importances[0:10].plot.barh(color = 'skyblue')
plt.xlabel(None)
plt.ylabel(None)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
# plt.savefig('plots/LGBM_feature_imp', dpi = 150)





plt.title('LGBM')
# plt.savefig('LGBM_feature_importances', dpi = 300)


"""RandomizedSearchCV """
from sklearn.model_selection import RandomizedSearchCV

grid_values_LGBM = {'n_estimators': list(np.arange(100, 1100, 100)), 'learning_rate': [0.01], 'max_depth': list(np.arange(1, 26, 2)), 'max_bin': list(np.arange(100, 600, 100)), 'num_leaves': list(np.arange(100, 800, 100))}

fit_params_LGBM={"eval_metric" : "auc", 
            "eval_set" : [[X_valid_encoded4_scaled, y_valid]]}

LGBMclf2 = lgb.LGBMClassifier(eval_metric="auc", n_jobs = 8)

LGBMclf_search = RandomizedSearchCV(LGBMclf2, param_distributions = grid_values_LGBM, scoring = 'roc_auc', cv = 3, verbose = 1, n_iter = 10)
LGBMclf_search.fit(X_train_encoded4_scaled, y_train, **fit_params_LGBM)
print('Grid best parameters (roc_auc): ', LGBMclf_search.best_params_)
# Grid best parameters (roc_auc):  {'n_estimators': 900, 'max_depth': 7, 'learning_rate': 0.01}


#%% all ROC plots




fpr_knnclf = np.load('ROC/fpr_knnclf.npy')
tpr_knnclf = np.load('ROC/tpr_knnclf.npy')
auc_knnclf = np.load('ROC/auc_knnclf.npy')

fpr_XGBC_model = np.load('ROC/fpr_XGBC_model.npy')
tpr_XGBC_model = np.load('ROC/tpr_XGBC_model.npy')
auc_XGBC_model = np.load('ROC/auc_XGBC_model.npy')


fpr_RFC = np.load('ROC/fpr_RFC.npy')
tpr_RFC = np.load('ROC/tpr_RFC.npy')
auc_RFC = np.load('ROC/auc_RFC.npy')


fpr_LGBMclf = np.load('ROC/fpr_LGBMclf.npy')
tpr_LGBMclf = np.load('ROC/tpr_LGBMclf.npy')
auc_LGBMclf = np.load('ROC/auc_LGBMclf.npy')


fpr_nn = np.load('ROC/fpr_nn.npy')
tpr_nn = np.load('ROC/tpr_nn.npy')
auc_nn = np.load('ROC/auc_nn.npy')



plt.figure(figsize = (6, 6), dpi = 150)
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_knnclf, tpr_knnclf, lw=2, label='KNN classifier (k = 7): area = {:0.3f}'.format(auc_knnclf), color='skyblue')
plt.plot(fpr_XGBC_model, tpr_XGBC_model, lw=2, label='XGBoost classifier: area = {:0.3f}'.format(auc_XGBC_model), color='violet')
plt.plot(fpr_RFC, tpr_RFC, lw=2, label='Random Forest classifier: area = {:0.3f}'.format(auc_RFC), color='forestgreen')
plt.plot(fpr_LGBMclf, tpr_LGBMclf, lw=2, label='LightGBM classifier: area = {:0.3f}'.format(auc_LGBMclf), color='gold')
plt.plot(fpr_nn, tpr_nn, lw=2, label='Neural Network: area = {:0.3f}'.format(auc_nn), color='dimgray')
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve', fontsize=16)
plt.legend(loc='lower right', fontsize=10)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.tight_layout()
# plt.savefig('plots/roc_all', dpi = 150)

















