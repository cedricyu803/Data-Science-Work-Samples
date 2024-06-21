# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:00:00 2021

@author: Cedric Yu
"""

# import matplotlib.pyplot as plt
# import math
# import re


"""
#####################################

# https://www.kaggle.com/c/word2vec-nlp-tutorial/overview/description

In this tutorial competition, we dig a little "deeper" into sentiment analysis. Google's Word2Vec (Sequence Model course 2.6-7) is a deep-learning inspired method that focuses on the meaning of words. Word2Vec attempts to understand meaning and semantic relationships among words. It works in a way that is similar to deep approaches, such as recurrent neural nets or deep neural nets, but is computationally more efficient. This tutorial focuses on Word2Vec for sentiment analysis.

Sentiment analysis is a challenging subject in machine learning. People express their emotions in language that is often obscured by sarcasm, ambiguity, and plays on words, all of which could be very misleading for both humans and computers. 

We consider an IMDB sentiment analysis data set, which has 100,000 multi-paragraph movie reviews, both positive and negative. 

#####################################

# Metric
Submissions are judged on [area under the ROC curve]. 

Submission Instructions
You should submit a comma-separated file with 25,000 row plus a header row. There should be 2 columns: "id" and "sentiment", which contain your binary predictions: 1 for positive reviews, 0 for negative reviews. For an example, see "sampleSubmission.csv" on the Data page. 

id,sentiment
123_45,0 
678_90,1
12_34,0
...

#####################################

#Dataset

The labeled data set consists of 50,000 IMDB movie reviews, specially selected for sentiment analysis. The sentiment of reviews is binary, meaning the IMDB rating < 5 results in a sentiment score of 0, and rating >=7 have a sentiment score of 1. No individual movie has more than 30 reviews. 
The 25,000 review labeled training set does not include any of the same movies as the 25,000 review test set. In addition, there are another 50,000 IMDB reviews provided without any rating labels.

File descriptions

# labeledTrainData - The [labeled training set]. The file is [tab-delimited] and has a [header row] followed by 25,000 rows containing an id, sentiment, and text for each review.  

# testData - The [test set]. The tab-delimited file has a header row followed by 25,000 rows containing an id and text for each review. [Your task is to predict the sentiment for each one.]

# unlabeledTrainData - An extra training set with no labels. The tab-delimited file has a header row followed by 50,000 rows containing an id and text for each review. 

# sampleSubmission - A comma-delimited sample submission file in the correct format.
Data fields
id - Unique ID of each review
sentiment - Sentiment of the review; 1 for positive reviews and 0 for negative reviews
review - Text of the review
"""

# %% This file

"""
First trial: without using embedding vectors
uses CountVectorizer and TfidfVectorizer from nltk

"""

# %% Workflow

"""
# load datasets and clean text
    # get rid of html syntax using bs4.BeautifulSoup
    # get rid of unimportant punctuation marks including apostrophe
    # lower case
    # lemmatise with nltk.WordNetLemmatizer()
    # tokenise with nltk.word_tokenize()

# train-validation split
# uses CountVectorizer/TfidfVectorizer from nltk, with stop_words='english', ngram_range = (1,5), and experiments with max_df and min_df
# LogisticRegression, MultinomialNB, XGBClassifier
# predict

"""


# %% Preamble

# Make the output look better
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
import re
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
# default='warn' # ignores warning about dropping columns inplace
pd.options.mode.chained_assignment = None
# import re

os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\8_nlp_imdb_sentiment_classification')

# %% load dataset labeledTrainData.tsv and testData.tsv

labeledTrainData = pd.read_csv("labeledTrainData.tsv", sep='\t', header=[0])
testData = pd.read_csv("testData.tsv", sep='\t', header=[0])


# %% preprocessing, tokenisation and padding


# for getting rid of html syntax
# Import BeautifulSoup into your workspace

"""preprocess text by removing html syntax, unimportant punctuation marks (e.g. ..., --), lowering case, and lemmatizing. Then tokenise and pad to max_len"""

WNlemma_n = nltk.WordNetLemmatizer()


def text_preprocess(text, pad=False, max_len=3000):
    # remove html syntax
    text1 = BeautifulSoup(text).get_text()
    # get rid of unimportant punctuation marks
    # !!! get rid of apostrophe too (and single quotation marks)
    # text1 = re.sub(r'([^\d\w\'\s\.\-]+|[-\.]{2,})', ' ', text1)
    # text1 = re.sub(r'([^\d\w\s\.\-]+|[-\.]{2,})', ' ', text1)
    # only keep alphabets and apostrophe
    text1 = re.sub(r'[^a-zA-Z\s]+', ' ', text1)
    # lower case
    text1 = text1.lower()
    # lemmatise
    text1 = WNlemma_n.lemmatize(text1)
    # tokenise
    text1 = nltk.word_tokenize(text1)
    stop_words = set(stopwords.words('english'))
    text1 = [word for word in text1 if not word in stop_words]
    if pad == True:
        # pad to max_len by '-1 empty'
        if len(text1) < max_len:
            text1 += ['-1 empty' for i in range(max_len - len(text1))]
    return text1


def pd_text_preprocess(row, pad=False, max_len=3000):
    row['review_preprocessed'] = text_preprocess(
        row['review'], pad=pad, max_len=max_len)
    row['length'] = len(row['review_preprocessed'])
    return row


# no padding
labeledTrainData = labeledTrainData.apply(pd_text_preprocess, axis=1)

testData = testData.apply(pd_text_preprocess, axis=1)
X_test = testData['review_preprocessed']

# %% train-validation split


X_train, X_valid, y_train, y_valid = train_test_split(
    labeledTrainData['review_preprocessed'], labeledTrainData['sentiment'], random_state=0, train_size=0.8)

# %% baseline models with CountVectorizer, tfidf, ngrams, and logistic regression
"""No embedding vector"""

"""# CountVectorizer"""


"""# grid search"""
# min_df_list = np.arange(0,4,1)
# max_df_list = np.arange(0.2,0.9,0.2)
# AUC_count = []


# for min_df in min_df_list:
#     for max_df in max_df_list:
#         # Fit the CountVectorizer to the training data
#         # minimum document frequency of 10 and max_df = 0.2, ignore stop words, ngram_range = (1,5)
#         vect_count0 = CountVectorizer(min_df=min_df, max_df=max_df, stop_words='english', ngram_range = (1,5)).fit(X_train)
#         # vect_count.get_feature_names()[::2000]
#         X_train_count_vectorised0 = vect_count0.transform(X_train)

#         # logistic regression
#         model_count0 = LogisticRegression(max_iter = 10000)
#         model_count0.fit(X_train_count_vectorised0, y_train)

#         # Predict the transformed test documents
#         predictions_count0 = model_count0.predict(vect_count0.transform(X_valid))
#         AUC_count.append((min_df, max_df, roc_auc_score(y_valid, predictions_count0)))


# Fit the CountVectorizer to the training data
# minimum document frequency of 10 and max_df = 0.2, ignore stop words, ngram_range = (1,5)
vect_count = CountVectorizer(
    min_df=0, max_df=0.2, stop_words='english', ngram_range=(1, 5)).fit(X_train)
# vect_count.get_feature_names()[::2000]
X_train_count_vectorised = vect_count.transform(X_train)
X_valid_count_vectorised = vect_count.transform(X_valid)


#################################

# logistic regression

model_count = LogisticRegression(max_iter=10000)
model_count.fit(X_train_count_vectorised, y_train)

# Predict the transformed test documents
predictions_count = model_count.predict(X_valid_count_vectorised)

print('AUC: ', roc_auc_score(y_valid, predictions_count))
# removed most punctuation marks
# AUC:  0.8875348608511041
# kept only alphabets, dropped stopwords
# AUC:  0.8833064220794353

# get the feature names as numpy array
feature_names_count = np.array(vect_count.get_feature_names())
# Sort the coefficients from the model
sorted_coef_index_count = model_count.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1]
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}'.format(
    feature_names_count[sorted_coef_index_count[:10]]))
print('Largest Coefs: \n{}'.format(
    feature_names_count[sorted_coef_index_count[:-11:-1]]))
# Smallest Coefs:
# ['worst' 'awful' 'waste' 'boring' 'worse' 'terrible' 'poor' 'dull'
#  'horrible' 'avoid']
# Largest Coefs:
# ['excellent' 'wonderful' 'perfect' 'favorite' 'amazing' 'best' 'enjoyed'
#  'loved' 'fantastic' 'today']

model_count_pred = model_count.predict(vect_count.transform(X_test))
model_count_pred = pd.Series(
    model_count_pred, index=testData['id'], name='sentiment')
model_count_pred.to_csv('y_test_pred_count_logreg.csv')

#################################


MNB_clf_count = MultinomialNB(alpha=0.1)
MNB_clf_count.fit(X_train_count_vectorised, y_train)


print('AUC: ', roc_auc_score(
    y_valid, MNB_clf_count.predict(X_valid_count_vectorised)))
# removed most punctuation marks
# AUC:  0.877616324481857
# kept only alphabets, dropped stopwords
# AUC:  0.8758732819266494


#################################


XGBC_model_count = XGBClassifier(eval_metric='auc', n_jobs=6)
XGBC_model_count.fit(X_train_count_vectorised, y_train, eval_set=[(
    X_train_count_vectorised, y_train), (X_valid_count_vectorised, y_valid)], early_stopping_rounds=40)

print('AUC: ', roc_auc_score(
    y_valid, XGBC_model_count.predict(X_valid_count_vectorised)))
# AUC:  0.85215221739342

XGBC_model_count_pred = XGBC_model_count.predict(vect_count.transform(X_test))
XGBC_model_count_pred = pd.Series(
    XGBC_model_count_pred, index=testData['id'], name='sentiment')
XGBC_model_count_pred.to_csv('y_test_pred_count_XGBC.csv')


#################################
"""# TfidfVectorizer"""


# min_df_list = np.arange(0,6,2)
# max_df_list = np.arange(0.2,0.6,0.2)
# AUC_tfidf = []


# for min_df in min_df_list:
#     for max_df in max_df_list:
#         # Fit the TfidfVectorizer to the training data
#         # minimum document frequency of 10 and max_df = 0.2, ignore stop words, ngram_range = (1,3)
#         vect_tfidf0 = TfidfVectorizer(min_df=min_df, max_df=max_df, stop_words='english', ngram_range = (1,5)).fit(X_train)
#         # vect_tfidf.get_feature_names()[::2000]
#         X_train_tfidf_vectorised0 = vect_tfidf0.transform(X_train)

#         # logistic regression
#         model_tfidf0 = LogisticRegression(max_iter = 10000)
#         model_tfidf0.fit(X_train_tfidf_vectorised0, y_train)

#         # Predict the transformed test documents
#         predictions_tfidf0 = model_tfidf0.predict(vect_tfidf0.transform(X_valid))
#         AUC_tfidf.append((min_df, max_df, roc_auc_score(y_valid, predictions_tfidf0)))


# Fit the TfidfVectorizer
# minimum document frequency of 4 and max_df = 0.2, ignore stop words, ngram_range = (1,3)
vect_tfidf = TfidfVectorizer(
    min_df=4, max_df=0.2, stop_words='english', ngram_range=(1, 5)).fit(X_train)
# vect_tfidf.get_feature_names()[::2000]
X_train_tfidf_vectorised = vect_tfidf.transform(X_train)
X_valid_tfidf_vectorised = vect_tfidf.transform(X_valid)

#################################

# logistic regression

model_tfidf = LogisticRegression(max_iter=10000)
model_tfidf.fit(X_train_tfidf_vectorised, y_train)

# Predict the transformed test documents
predictions_tfidf = model_tfidf.predict(vect_tfidf.transform(X_valid))

print('AUC: ', roc_auc_score(y_valid, predictions_tfidf))
# removed most punctuation marks
# AUC:  0.8938911240239601
# kept only alphabets, dropped stopwords
# AUC:  0.8911208227801097

# get the feature names as numpy array
feature_names_tfidf = np.array(vect_tfidf.get_feature_names())
# Sort the coefficients from the model
sorted_coef_index_tfidf = model_tfidf.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1]
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}'.format(
    feature_names_tfidf[sorted_coef_index_tfidf[:10]]))
print('Largest Coefs: \n{}'.format(
    feature_names_tfidf[sorted_coef_index_tfidf[:-11:-1]]))
# Smallest Coefs:
# ['worst' 'awful' 'waste' 'boring' 'worse' 'poor' 'terrible' 'horrible'
#  'script' 'supposed']
# Largest Coefs:
# ['excellent' 'best' 'wonderful' 'perfect' 'love' 'amazing' 'favorite'
#  'loved' 'today' 'enjoyed']


model_tfidf_pred = model_tfidf.predict(vect_tfidf.transform(X_test))
model_tfidf_pred = pd.Series(
    model_tfidf_pred, index=testData['id'], name='sentiment')
model_tfidf_pred.to_csv('y_test_pred_tfidf_logreg.csv')


#################################


MNB_clf_tfidf = MultinomialNB(alpha=0.1)
MNB_clf_tfidf.fit(X_train_tfidf_vectorised, y_train)


print('AUC: ', roc_auc_score(
    y_valid, MNB_clf_tfidf.predict(X_valid_tfidf_vectorised)))
# AUC:  0.8828371290792637

MNB_tfidf_pred = MNB_clf_tfidf.predict(vect_tfidf.transform(X_test))
MNB_tfidf_pred = pd.Series(
    MNB_tfidf_pred, index=testData['id'], name='sentiment')
MNB_tfidf_pred.to_csv('y_test_pred_tfidf_MNB.csv')

#################################


XGBC_model_tfidf = XGBClassifier(eval_metric='auc', n_jobs=6)
XGBC_model_tfidf.fit(X_train_tfidf_vectorised, y_train, eval_set=[(
    X_train_tfidf_vectorised, y_train), (X_valid_tfidf_vectorised, y_valid)], early_stopping_rounds=40)

print('AUC: ', roc_auc_score(
    y_valid, XGBC_model_tfidf.predict(X_valid_tfidf_vectorised)))
# AUC:  0.8456653460731763

XGBC_tfidf_pred = XGBC_model_tfidf.predict(vect_tfidf.transform(X_test))
XGBC_tfidf_pred = pd.Series(
    XGBC_tfidf_pred, index=testData['id'], name='sentiment')
XGBC_tfidf_pred.to_csv('y_test_pred_tfidf_XGBC.csv')
