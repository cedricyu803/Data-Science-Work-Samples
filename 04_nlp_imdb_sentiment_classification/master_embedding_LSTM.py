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

# %% This file

"""
uses embedding vector from GloVe 6B.50d
train an LSTM neural network
"""

# %% Workflow

"""
# load datasets and clean text
    # get rid of html syntax using bs4.BeautifulSoup
    # get rid of unimportant punctuation marks including apostrophe
    # lower case
    # lemmatise with nltk.WordNetLemmatizer()
    # tokenise with nltk.word_tokenize()
    # (remove stop words)
    # set maximum sequence length max_len = 2700 (max length in datasets =2602) and pad with '-1 empty'

# train-validation split
# load GloVe vectors
# map words in pre-processed datasets to GloVe indices, save to files 
    # unknown/padded words are set to index = 0, which will be mapped to zero embedding vectors
# define nn model: embedding --> Bidirectional LSTM --> Bidirectional LSTM --> Dropout --> Fully Connected with tanh --> output with sigmoid
# optimise with Adam
# predict

"""


# %% Preamble

# Make the output look better
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.initializers import glorot_uniform
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional
from keras.models import Model
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


# %% load embedding matrix pre-trained using glove

# https://nlp.stanford.edu/projects/glove/
# read in the glove file. Each line in the text file is a string containing a word followed by the embedding vector, all separated by a whitespace
# word_to_vec_map is a dict of words to their embedding vectors
# (400,000 words, with the valid indices starting from 1 to 400,000
def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        # starts from 1
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


# index is valued  1-40000
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(
    'glove.6B.50d.txt')


# %% functions for preprocessing, tokenisation and padding


# for getting rid of html syntax

"""preprocess text by removing html syntax, unimportant punctuation marks (e.g. ..., --), lowering case, and lemmatizing. Then tokenise and pad to max_len"""

WNlemma_n = nltk.WordNetLemmatizer()


def text_preprocess(text, pad=False, max_len=3000):
    # remove html syntax
    text1 = BeautifulSoup(text).get_text()
    # get rid of unimportant punctuation marks
    # !!! get rid of apostrophe too (and single quotation marks)
    # text1 = re.sub(r'([^\d\w\'\s\.\-]+|[-\.]{2,})', ' ', text1)
    text1 = re.sub(r'([^\d\w\s\.\-]+|[-\.]{2,})', ' ', text1)
    # only keep alphabets and apostrophe
    # text1 = re.sub(r'[^a-zA-Z_\'\s]+', ' ', text1)
    # lower case
    text1 = text1.lower()
    # lemmatise
    text1 = WNlemma_n.lemmatize(text1)
    # tokenise
    text1 = nltk.word_tokenize(text1)
    if pad == True:
        # pad to max_len by '-1 empty'
        if len(text1) < max_len:
            text1 += ['-1 empty' for i in range(max_len - len(text1))]
    return text1


def pd_text_preprocess(row, pad=False, max_len=3000):
    row['review_preprocessed'] = text_preprocess(
        row['review'], pad=True, max_len=max_len)
    # row['length'] = len(row['review_preprocessed'])
    return row


# %% function that converts training sentences into indices with padding
# set unknown and padded tokens to 0; the embedding vectors will be zero-vectors
padded_token_index = 0
unknown_token_index = 0


def sentences_to_indices(X, word_to_index):
    """
    Converts an array of [padded, tokenised sentences] into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to 'Embedding()'

    Arguments:
    X -- array of padded, tokenised sentences, of shape (m, max_len)
    word_to_index -- a dictionary containing the each word mapped to its index

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    # set padded tokens to index = padded_token_index
    word_to_index0 = word_to_index.copy()
    word_to_index0['-1 empty'] = padded_token_index

    # set unknown tokens to index = unknown_token_index
    def word_to_index00(key):
        return word_to_index0.get(key, unknown_token_index)

    # speed up computation with vectorisation
    X_indices = np.vectorize(word_to_index00)(X)

    return X_indices

# test
# import time
# start = time.time()
# sentences_to_indices(X_train[0:10], word_to_index)
# end = time.time()
# print(end - start)
# X1_indices = sentences_to_indices(X1, word_to_index)
# # array([[292526.,  58997., 264550., ...,      0.,      0.,      0.],
    # [358160., 251034., 174032., ...,      0.,      0.,      0.]])


# %% LSTM with embedding layer


"""
Overview of model:
Embedding ---> Bidirectional LSTM ---> Bidirectional LSTM ---> Dropout ---> Fully Connected with tanh ---> output with sigmoid
"""


"""
2. The Embedding layer

# In Keras, the embedding matrix is represented as a "layer".
# The embedding matrix maps word indices to embedding vectors.
    # The word indices are [positive] integers.
    # The embedding vectors are dense vectors of fixed size.
    # When we say a vector is "dense", in this context, it means that most of the values are non-zero. As a counter-example, a one-hot encoded vector is not "dense."
# The embedding matrix can be derived in two ways:
    # Training a model to derive the embeddings from scratch.
    # Using a pretrained embedding

# Using and updating pre-trained embeddings
# In this part, you will learn how to create an Embedding() layer in Keras
# You will initialize the Embedding layer with the GloVe 50-dimensional vectors.
# In the code below, we'll show you how Keras allows you to either train or leave fixed this layer.
# Because our training set is quite small, we will leave the GloVe embeddings fixed instead of updating them.
"""

"""build an embedding layer"""


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    # Embedding() in Keras:
    # input_dim: Integer. Size of the vocabulary, i.e. maximum integer index + 1.
    vocab_len = len(word_to_index) + 1
    # output_dim in Embedding()
    # dimensionality of GloVe word vectors (= 50)
    emb_dim = list(word_to_vec_map.values())[0].shape[0]

    # using word_to_vec_map from pre-trained Glove, construct the embedding matrix
    # Initialize the embedding matrix as a numpy array of zeros.
    embedding_matrix = np.zeros([vocab_len, emb_dim])
    # Set each row "idx" of the embedding matrix to be the word vector representation of the idx-th word of the vocabulary
    # idx starts from 1 to 40000, so the first row of embedding_matrix is zero representing unknown words
    for word, idx in word_to_index.items():
        embedding_matrix[idx, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct input and output sizes
    # Make it non-trainable.
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

    # !!! Weights are created when the Model is first called on inputs or build() is called with an input_shape.
    # somehow embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix) does not work...
    # in our model, embedding_layer takes the input layer with input shape (max_len,)
    # here embedding_layer does not know max_len yet, so use (None,)
    embedding_layer.build((None,))
    embedding_layer.set_weights([embedding_matrix])

    return embedding_layer


# embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
# print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])
# weights[0][1][3] = -0.3403


"""
3. Building an LSTM model

The model takes as input an array of sentences of shape (m, max_len, ) defined by input_shape.
The model outputs a sigmoid probability vector of shape (m, 1).
"""


def sentiment_classification_model(input_shape, word_to_vec_map, word_to_index, num_classes=1):
    """

    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,000 words)

    Returns:
    model -- a model instance in Keras
    """

    # Define sentence_indices as the input of the graph
    # It should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(shape=input_shape, dtype='int32')

    # Create the embedding layer pretrained with GloVe Vectors
    embedding_layer = pretrained_embedding_layer(
        word_to_vec_map, word_to_index)

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)

    # !!! You must set return_sequences = True when stacking LSTM layers (except the last LSTM layer), so that the next LSTM layer has as input a sequence of length = num_timesteps
    X = Bidirectional(LSTM(256, return_sequences=True))(embeddings)
    # this is the last LSTM layer; it should only output the final state for the next (non-LSTM) layer
    X = Bidirectional(LSTM(256, return_sequences=False))(X)
    X = Dense(64, activation='tanh')(X)
    X = Dropout(.2)(X)
    X = Dense(num_classes)(X)
    if num_classes == 1:
        # Add a sigmoid activation
        X = Activation('sigmoid')(X)
    elif num_classes > 1:
        # Add a softmax activation
        X = Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)

    return model


# %% load datasets labeledTrainData.tsv and testData.tsv

labeledTrainData = pd.read_csv("labeledTrainData.tsv", sep='\t', header=[0])
testData = pd.read_csv("testData.tsv", sep='\t', header=[0])

"""pre-processing"""
max_len = 2700
labeledTrainData = labeledTrainData.apply(
    pd_text_preprocess, axis=1, pad=True, max_len=max_len)
# max length = 2602

# X_test is np.array of shape (m, max_len)
testData = testData.apply(pd_text_preprocess, axis=1,
                          pad=True, max_len=max_len)
X_test = np.array(testData['review_preprocessed'].tolist())
# max length = 2331

# %% train-validation split


X_train, X_valid, y_train, y_valid = train_test_split(
    labeledTrainData['review_preprocessed'], labeledTrainData['sentiment'], random_state=0, train_size=0.8)

# X_train and X_valid are np.array of shape (m, max_len)
X_train = np.array(X_train.tolist())
X_valid = np.array(X_valid.tolist())
y_train1 = np.expand_dims(y_train.to_numpy(), -1)
y_valid1 = np.expand_dims(y_valid.to_numpy(), -1)

X_train_indices = sentences_to_indices(X_train, word_to_index)
X_valid_indices = sentences_to_indices(X_valid, word_to_index)
X_test_indices = sentences_to_indices(X_test, word_to_index)

# X_train_indices.shape
# (20000, 2700)
# np.save('X_train_indices.npy', X_train_indices)
# np.save('X_valid_indices.npy', X_valid_indices)
# np.save('X_test_indices.npy', X_test_indices)
# np.save('y_train.npy', y_train)
# np.save('y_valid.npy', y_valid)

# %% load pre-processed datasets

# !!!
X_train_indices = np.load('X_train_indices.npy')
X_valid_indices = np.load('X_valid_indices.npy')
X_test_indices = np.load('X_test_indices.npy')
y_train = np.load('y_train.npy')
y_valid = np.load('y_valid.npy')
y_train1 = np.expand_dims(y_train, -1)
y_valid1 = np.expand_dims(y_valid, -1)

max_len = X_train_indices.shape[-1]
# 1500

# %% fit model

model = sentiment_classification_model(
    (max_len,), word_to_vec_map, word_to_index)
# model.summary()

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.0001), metrics=[tf.keras.metrics.AUC()])


callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

# input_shape = (m, max_len)
history = model.fit(X_train_indices, y_train1, validation_data=(
    X_valid_indices, y_valid1), epochs=100, batch_size=32, callbacks=[callback])

# Epoch 1/50
# 625/625 [==============================] - 466s 738ms/step - loss: 0.6678 - auc_2: 0.6382 - val_loss: 0.5679 - val_auc_2: 0.8130
# Epoch 2/50
# 625/625 [==============================] - 460s 736ms/step - loss: 0.4505 - auc_2: 0.8709 - val_loss: 0.3686 - val_auc_2: 0.9193
# Epoch 3/50
# 625/625 [==============================] - 471s 753ms/step - loss: 0.3548 - auc_2: 0.9224 - val_loss: 0.3263 - val_auc_2: 0.9361
# Epoch 4/50
# 625/625 [==============================] - 457s 732ms/step - loss: 0.3199 - auc_2: 0.9373 - val_loss: 0.3219 - val_auc_2: 0.9452
# Epoch 5/50
# 625/625 [==============================] - 453s 725ms/step - loss: 0.2895 - auc_2: 0.9488 - val_loss: 0.2906 - val_auc_2: 0.9498
# Epoch 6/50
# 625/625 [==============================] - 458s 733ms/step - loss: 0.2622 - auc_2: 0.9580 - val_loss: 0.3145 - val_auc_2: 0.9494
# Epoch 7/50
# 625/625 [==============================] - 477s 764ms/step - loss: 0.2334 - auc_2: 0.9666 - val_loss: 0.2981 - val_auc_2: 0.9522
# Epoch 8/50
# 625/625 [==============================] - 452s 724ms/step - loss: 0.1998 - auc_2: 0.9753 - val_loss: 0.2978 - val_auc_2: 0.9495
# Epoch 9/50
# 625/625 [==============================] - 454s 726ms/step - loss: 0.1603 - auc_2: 0.9834 - val_loss: 0.3248 - val_auc_2: 0.9484
# Epoch 10/50
# 625/625 [==============================] - 452s 724ms/step - loss: 0.1210 - auc_2: 0.9900 - val_loss: 0.4022 - val_auc_2: 0.9442


"""
tried 3 b-LSTM layers (128 units) and 2 FC layers (64, 16, tanh)
X = Bidirectional(LSTM(128, return_sequences = True))(embeddings)
    X = Bidirectional(LSTM(128, return_sequences = True))(X)
    X = Bidirectional(LSTM(128, return_sequences = False))(X)
    X = Dense(64, activation = 'tanh')(X)
    X = Dense(16, activation = 'tanh')(X)
    X = Dropout(.2)(X)
    X = Dense(1)(X)
    X = Activation('sigmoid')(X)
learning_rate=0.00005

625/625 [==============================] - 468s 737ms/step - loss: 0.6171 - auc: 0.7183 - val_loss: 0.5945 - val_auc: 0.7776
Epoch 2/200
625/625 [==============================] - 460s 736ms/step - loss: 0.5678 - auc: 0.7776 - val_loss: 0.5534 - val_auc: 0.8109
Epoch 3/200
625/625 [==============================] - 460s 736ms/step - loss: 0.5463 - auc: 0.7983 - val_loss: 0.5233 - val_auc: 0.8269
Epoch 4/200
625/625 [==============================] - 459s 735ms/step - loss: 0.5552 - auc: 0.7897 - val_loss: 0.5516 - val_auc: 0.8061
Epoch 5/200
625/625 [==============================] - 460s 737ms/step - loss: 0.5564 - auc: 0.7884 - val_loss: 0.5429 - val_auc: 0.8073
Epoch 6/200
625/625 [==============================] - 461s 737ms/step - loss: 0.5302 - auc: 0.8123 - val_loss: 0.5299 - val_auc: 0.8367
Epoch 7/200
625/625 [==============================] - 461s 738ms/step - loss: 0.5109 - auc: 0.8285 - val_loss: 0.4879 - val_auc: 0.8454
Epoch 8/200
625/625 [==============================] - 461s 738ms/step - loss: 0.5056 - auc: 0.8328 - val_loss: 0.5274 - val_auc: 0.8508
Epoch 9/200
625/625 [==============================] - 460s 736ms/step - loss: 0.4906 - auc: 0.8438 - val_loss: 0.5463 - val_auc: 0.8487
Epoch 10/200
625/625 [==============================] - 460s 736ms/step - loss: 0.5487 - auc: 0.7966 - val_loss: 0.5207 - val_auc: 0.8425
Epoch 11/200
625/625 [==============================] - 460s 736ms/step - loss: 0.4854 - auc: 0.8479 - val_loss: 0.4955 - val_auc: 0.8510
Epoch 12/200
625/625 [==============================] - 462s 739ms/step - loss: 0.4841 - auc: 0.8486 - val_loss: 0.4661 - val_auc: 0.8639
Epoch 13/200
625/625 [==============================] - 460s 736ms/step - loss: 0.4940 - auc: 0.8413 - val_loss: 0.6189 - val_auc: 0.7526
Epoch 14/200
625/625 [==============================] - 459s 734ms/step - loss: 0.5945 - auc: 0.7488 - val_loss: 0.5724 - val_auc: 0.7885
Epoch 15/200
625/625 [==============================] - 458s 733ms/step - loss: 0.4970 - auc: 0.8392 - val_loss: 0.4585 - val_auc: 0.8680
Epoch 16/200
625/625 [==============================] - 458s 733ms/step - loss: 0.4781 - auc: 0.8527 - val_loss: 0.4818 - val_auc: 0.8602
Epoch 17/200
625/625 [==============================] - 459s 735ms/step - loss: 0.4701 - auc: 0.8583 - val_loss: 0.4640 - val_auc: 0.8752
Epoch 18/200
625/625 [==============================] - 458s 733ms/step - loss: 0.4786 - auc: 0.8529 - val_loss: 0.4716 - val_auc: 0.8608
Epoch 19/200
625/625 [==============================] - 459s 735ms/step - loss: 0.4532 - auc: 0.8693 - val_loss: 0.4541 - val_auc: 0.8699
Epoch 20/200
625/625 [==============================] - 458s 734ms/step - loss: 0.4475 - auc: 0.8731 - val_loss: 0.4677 - val_auc: 0.8774
Epoch 21/200
625/625 [==============================] - 459s 734ms/step - loss: 0.4434 - auc: 0.8749 - val_loss: 0.4375 - val_auc: 0.8840
Epoch 22/200
625/625 [==============================] - 459s 734ms/step - loss: 0.4295 - auc: 0.8835 - val_loss: 0.4389 - val_auc: 0.8827
Epoch 23/200
625/625 [==============================] - 458s 733ms/step - loss: 0.4253 - auc: 0.8859 - val_loss: 0.4175 - val_auc: 0.8925
Epoch 24/200
625/625 [==============================] - 458s 733ms/step - loss: 0.4195 - auc: 0.8896 - val_loss: 0.4081 - val_auc: 0.8970
Epoch 25/200
625/625 [==============================] - 459s 735ms/step - loss: 0.4129 - auc: 0.8931 - val_loss: 0.4510 - val_auc: 0.8972

no significant gain
"""


history_df = pd.DataFrame(history.history)
history_df.columns

plt.figure()
plt.plot(history.history['loss'], color='red', label='loss')
plt.plot(history.history['val_loss'], color='blue', label='val_loss')
plt.legend()
# plt.savefig('LSTM_2_loss', dpi = 150)

plt.figure()
plt.plot(history.history['auc_1'], color='red', label='auc')
plt.plot(history.history['val_auc_1'], color='blue', label='val_auc')
plt.legend()
# plt.savefig('LSTM_2_auc', dpi = 150)


"""# save the model"""

model.save('LSTM_model_2.h5')


prediction = model.predict(X_test_indices).squeeze()
prediction_bool = (prediction > 0.5).astype(int)
testData = pd.read_csv("testData.tsv", sep='\t', header=[0])
LSTM_embedding_pred = pd.Series(
    prediction_bool, index=testData['id'], name='sentiment')
LSTM_embedding_pred.to_csv('LSTM_embedding_no_stopwords_pred.csv')

# with stopwords
# LSTM two layers (128)
# 0.85080
# LSTM_2 (256x2)
# 0.87644
# without stopwords
# LSTM_2 (256x2)
# 0.83696
