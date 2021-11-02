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

# Dataset

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

#%% This file

"""
uses BertForSequenceClassification and pre-trained bert-base-uncased from hugging face transformers
https://huggingface.co/bert-base-uncased
"""

#%% Workflow

"""
# load datasets and clean text
    # get rid of html syntax using bs4.BeautifulSoup
    # keep alphabets and full stops
    # lower case

# train-validation split
# create indices for words
# uses BertForSequenceClassification and pre-trained bert-base-uncased from hugging face transformers, and re-train with our training data
# predict

"""


#%% Preamble

import pandas as pd
# Make the output look better
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None  # default='warn' # ignores warning about dropping columns inplace
import numpy as np
import matplotlib.pyplot as plt
# import re
import seaborn as sns

import os
os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\8_nlp_imdb_sentiment_classification')


#%% functions for preprocessing, tokenisation and padding

import re
import nltk
# from nltk.corpus import stopwords

# for getting rid of html syntax
from bs4 import BeautifulSoup    

"""preprocess text by removing html syntax, unimportant punctuation marks (e.g. ..., --), lowering case, and lemmatizing. Then tokenise and pad to max_len"""

# WNlemma_n = nltk.WordNetLemmatizer()

def text_preprocess(text, pad = False, max_len=3000):
    # remove html syntax
    text1 = BeautifulSoup(text).get_text() 
    # get rid of unimportant punctuation marks
    # !!! get rid of apostrophe too (and single quotation marks)
    # text1 = re.sub(r'([^\d\w\'\s\.\-]+|[-\.]{2,})', ' ', text1)
    # text1 = re.sub(r'([^\d\w\s\.\-]+|[-\.]{2,})', ' ', text1)
    # only keep alphabets
    text1 = re.sub(r'[^a-zA-Z\s]+', ' ', text1)
    # lower case
    text1 = text1.lower()
    # lemmatise
    # text1 = WNlemma_n.lemmatize(text1)
    # tokenise
    # text1 = nltk.word_tokenize(text1)
    # stop_words = set(stopwords.words('english'))
    # text1 = [word for word in text1 if not word in stop_words]
    # if pad == True:
    #     # pad to max_len by '-1 empty'
    #     if len(text1) < max_len:
    #         text1 += ['-1 empty' for i in range(max_len - len(text1))]
    return text1


def pd_text_preprocess(row, pad = False, max_len=3000):
    row['review_preprocessed'] = text_preprocess(row['review'], pad = True, max_len = max_len)
    # row['length'] = len(row['review_preprocessed'])
    return row


#%% load datasets labeledTrainData.tsv and testData.tsv

labeledTrainData = pd.read_csv ("labeledTrainData.tsv", sep = '\t', header = [0])
testData = pd.read_csv ("testData.tsv", sep = '\t', header = [0])

"""pre-processing"""
labeledTrainData = labeledTrainData.apply(pd_text_preprocess, axis = 1)

# X_test is np.array of shape (m, max_len)
testData = testData.apply(pd_text_preprocess, axis = 1)
X_test = testData['review_preprocessed']


#%% train-validation split

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(labeledTrainData['review_preprocessed'], labeledTrainData['sentiment'], random_state=0, train_size = 0.8)

y_train1 = y_train.to_list()
y_valid1 = y_valid.to_list()

#%% tokenisation and target label alignment with the huggingface library

"""
Before feeding the texts to a Transformer model, we will need to tokenize our input using a huggingface Transformer tokenizer. 
!!! It is crucial that the tokenizer we use must match the Transformer model type you are using
we will use the huggingface BertTokenizerFast tokenizer, which standardizes the length of our sequence to 512 and pads with zeros. Notice this matches the maximum length we used when creating tags.
"""

"""
Transformer models are often trained by tokenizers that split words into subwords. 
For instance, the word 'Africa' might get split into multiple subtokens. 
This can create some misalignment between [the list of target labels from the labeled dataset] and [the list of labels generated by the tokenizer], since the tokenizer can split one word into several, or add special tokens. 
Before processing, it is important that we align the lists of target labels and the list of labels generated by the selected tokenizer.

Our problem is many-to-one; the target of each instance is one label, so we can forgo the trouble of aligning labels of sub-words.
All we need to do is to tokenise and pad sentences
"""

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('pre-trained-transformer-bert-base-uncased/')


max_len = 512

X_train_tokenized = tokenizer(X_train.values.tolist(), truncation=True, is_split_into_words=False, padding='max_length', max_length=max_len)
X_train_tokenized_ids = X_train_tokenized['input_ids']


X_valid_tokenized = tokenizer(X_valid.values.tolist(), truncation=True, is_split_into_words=False, padding='max_length', max_length=max_len)
X_valid_tokenized_ids = X_valid_tokenized['input_ids']


X_test_tokenized = tokenizer(X_test.values.tolist(), truncation=True, is_split_into_words=False, padding='max_length', max_length=max_len)
X_test_tokenized_ids = X_test_tokenized['input_ids']


"""
examples
"""

# tokenizer.tokenize('hello there')
# # ['hello', 'there']
# tokenizer('hello there', truncation=True, is_split_into_words=False, padding='max_length', max_length=10)
# # {'input_ids': [101, 7592, 2045, 102, 0, 0, 0, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]}
# # each subword has an 'input_id': 101 is start of string and 102 is end of string
# tokenizer('hello there', truncation=True, is_split_into_words=False, padding='max_length', max_length=10).word_ids()
# # [None, 0, 1, None, None, None, None, None, None, None]
# # word_ids assigns, in ascending order, the same word id for all sub-words of the same original word

# example1 = labeledTrainData['review_preprocessed'].iloc[:1].values.tolist()[0]
# np.array(tokenizer.tokenize(example1))
# # array(['with', 'all', 'this', 'stuff', 'going', 'down', 'at', 'the',
# #        'moment', 'with', 'm', '##j', 'i', 've', 'started', 'listening',
# #        'to', 'his', 'music', 'watching', 'the', 'odd', 'documentary',
# #        'here', 'and', 'there', 'watched', 'the', 'wi', '##z', 'and',
# #        'watched', 'moon', '##walker', ..., 'most',
# #        'sick', '##est', 'liar', '##s', 'i', 'hope', 'he', 'is', 'not',
# #        'the', 'latter'], dtype='<U11')
# # so it splits words into subwords

# tokenizer(example1, truncation=True, is_split_into_words=False, padding='max_length', max_length=20)
# # {'input_ids': [101, 2007, 2035, 2023, 4933, 2183, 2091, 2012, 1996, 2617, 2007, 1049, 3501, 1045, 2310, 2318, 5962, 2000, 2010, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
# # each subword has an 'input_id': 101 is start of string and 102 is end of string
# tokenizer(example1, truncation=True, is_split_into_words=False, padding='max_length', max_length=20).word_ids()
# # [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, None]
# # word_ids assigns, in ascending order, the same word id for all sub-words of the same original word

# tokenizer(labeledTrainData['review_preprocessed'].iloc[0:4].values.tolist(), truncation=True, is_split_into_words=False, padding='max_length', max_length=20)

# # {'input_ids': [[101, 2007, 2035, 2023, 4933, 2183, 2091, 2012, 1996, 2617, 2007, 1049, 3501, 1045, 2310, 2318, 5962, 2000, 2010, 102], [101, 1996, 4438, 2162, 1997, 1996, 8484, 2011, 10805, 25445, 2003, 1037, 2200, 14036, 2143, 2008, 5525, 3632, 2000, 102], [101, 1996, 2143, 4627, 2007, 1037, 3208, 6141, 4330, 3228, 6160, 9387, 2728, 12385, 17190, 2638, 2000, 22289, 2380, 102], [101, 2009, 2442, 2022, 5071, 2008, 2216, 2040, 5868, 2023, 2143, 1996, 4602, 6361, 3850, 2412, 2134, 1056, 1045, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}

""" ---------------- end of example ----------------"""




#%% optimisation

"""
feed data into a pretrained 🤗 model. optimize a DistilBERT model, which matches the tokenizer used to preprocess your data
"""

import tensorflow as tf
from transformers import TFBertForSequenceClassification

Bert_trans_model = TFBertForSequenceClassification.from_pretrained('pre-trained-transformer-bert-base-uncased/', num_labels=1)

# load pre-trained(x2) weights
Bert_trans_model.load_weights('transformer_model_weights.h5')

# example2 = X_train_tokenized['input_ids'][0]

# label inputs are in a list [label1, label2, ...]
# Bert_trans_model.fit([example2, example2], [1,1])

# Bert_trans_model.predict([example2])
# # TFSequenceClassifierOutput(loss=None, logits=array([[0.30842814]], dtype=float32), hidden_states=None, attentions=None)

my_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
Bert_trans_model.compile(loss=my_loss, optimizer=optimizer, metrics=[tf.keras.metrics.AUC(from_logits=True)])

# callback = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss', patience=2, restore_best_weights=True)

# if you get GPU 'Resource exhausted: failed to allocate memory', reduce batch size
history = Bert_trans_model.fit(X_train_tokenized_ids, y_train1, validation_data = (X_valid_tokenized_ids, y_valid1), epochs = 2, batch_size = 4)
# Epoch 1/2
# 5000/5000 [==============================] - 1641s 326ms/step - loss: 0.2711 - auc: 0.9547 - val_loss: 0.1858 - val_auc: 0.9787
# Epoch 2/2
# 5000/5000 [==============================] - 1630s 326ms/step - loss: 0.1373 - auc: 0.9872 - val_loss: 0.2475 - val_auc: 0.9757

"""# save the model"""

# Bert_trans_model.save_weights('transformer_model_weights.h5')


#%% predict

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

prediction = Bert_trans_model.predict(X_test_tokenized_ids).logits    # logits
prediction = sigmoid(prediction)    #  apply sigmoid
prediction_bool = (prediction > 0.5).astype(int)
prediction_bool = prediction_bool.squeeze()
testData = pd.read_csv ("testData.tsv", sep = '\t', header = [0])
transformer_pred = pd.Series(prediction_bool, index = testData['id'], name = 'sentiment')
transformer_pred.to_csv('transformer_pred.csv')


# 0.92304  =)



# tf.keras.backend.clear_session()

# from numba import cuda 
# device = cuda.get_current_device()
# device.reset()

