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
exploratory data analysis

experiment with text cleaning strategies

"""


# %% Preamble

# Make the output look better
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

labeledTrainData0 = labeledTrainData.copy()

# %% preprocessing and tokenisation


labeledTrainData['sentiment'].mean()
# 0.5
# balanced class

# html syntax
labeledTrainData['review'].iloc[3]
# 'It must be assumed that those who praised this film (\\the greatest filmed opera ever,\\" didn\'t I read somewhere?) either don\'t care for opera, don\'t care for Wagner, or don\'t care about anything except their desire to appear Cultured. Either as a representation of Wagner\'s swan-song, or as a movie, this strikes me as an unmitigated disaster, with a leaden reading of the score matched to a tricksy, lugubrious realisation of the text.<br /><br />It\'s questionable that people with ideas as to what an opera (or, for that matter, a play, especially one by Shakespeare) is \\"about\\" should be allowed anywhere near a theatre or film studio; Syberberg, very fashionably, but without the smallest justification from Wagner\'s text, decided that Parsifal is \\"about\\" bisexual integration, so that the title character, in the latter stages, transmutes into a kind of beatnik babe, though one who continues to sing high tenor -- few if any of the actors in the film are the singers, and we get a double dose of Armin Jordan, the conductor, who is seen as the face (but not heard as the voice) of Amfortas, and also appears monstrously in double exposure as a kind of Batonzilla or Conductor Who Ate Monsalvat during the playing of the Good Friday music -- in which, by the way, the transcendant loveliness of nature is represented by a scattering of shopworn and flaccid crocuses stuck in ill-laid turf, an expedient which baffles me. In the theatre we sometimes have to piece out such imperfections with our thoughts, but I can\'t think why Syberberg couldn\'t splice in, for Parsifal and Gurnemanz, mountain pasture as lush as was provided for Julie Andrews in Sound of Music...<br /><br />The sound is hard to endure, the high voices and the trumpets in particular possessing an aural glare that adds another sort of fatigue to our impatience with the uninspired conducting and paralytic unfolding of the ritual. Someone in another review mentioned the 1951 Bayreuth recording, and Knappertsbusch, though his tempi are often very slow, had what Jordan altogether lacks, a sense of pulse, a feeling for the ebb and flow of the music -- and, after half a century, the orchestral sound in that set, in modern pressings, is still superior to this film."'

# get rid of html syntax
# Import BeautifulSoup into your workspace

example3 = BeautifulSoup(labeledTrainData['review'].iloc[3]).get_text()
# 'It must be assumed that those who praised this film (\\the greatest filmed opera ever,\\" didn\'t I read somewhere?) either don\'t care for opera, don\'t care for Wagner, or don\'t care about anything except their desire to appear Cultured. Either as a representation of Wagner\'s swan-song, or as a movie, this strikes me as an unmitigated disaster, with a leaden reading of the score matched to a tricksy, lugubrious realisation of the text.It\'s questionable that people with ideas as to what an opera (or, for that matter, a play, especially one by Shakespeare) is \\"about\\" should be allowed anywhere near a theatre or film studio; Syberberg, very fashionably, but without the smallest justification from Wagner\'s text, decided that Parsifal is \\"about\\" bisexual integration, so that the title character, in the latter stages, transmutes into a kind of beatnik babe, though one who continues to sing high tenor -- few if any of the actors in the film are the singers, and we get a double dose of Armin Jordan, the conductor, who is seen as the face (but not heard as the voice) of Amfortas, and also appears monstrously in double exposure as a kind of Batonzilla or Conductor Who Ate Monsalvat during the playing of the Good Friday music -- in which, by the way, the transcendant loveliness of nature is represented by a scattering of shopworn and flaccid crocuses stuck in ill-laid turf, an expedient which baffles me. In the theatre we sometimes have to piece out such imperfections with our thoughts, but I can\'t think why Syberberg couldn\'t splice in, for Parsifal and Gurnemanz, mountain pasture as lush as was provided for Julie Andrews in Sound of Music...The sound is hard to endure, the high voices and the trumpets in particular possessing an aural glare that adds another sort of fatigue to our impatience with the uninspired conducting and paralytic unfolding of the ritual. Someone in another review mentioned the 1951 Bayreuth recording, and Knappertsbusch, though his tempi are often very slow, had what Jordan altogether lacks, a sense of pulse, a feeling for the ebb and flow of the music -- and, after half a century, the orchestral sound in that set, in modern pressings, is still superior to this film."'

# get rid of unimportant punctuation marks
example3 = re.sub(r'([^\d\w\'\s\.\-]+|[-\.]{2,})', ' ', example3)
WNlemma_n = nltk.WordNetLemmatizer()

example3.lower()

print(nltk.word_tokenize(WNlemma_n.lemmatize(example3)))
['It', 'must', 'be', 'assumed', 'that', 'those', 'who', 'praised', 'this', 'film', 'the', 'greatest', 'filmed', 'opera', 'ever', 'did', "n't", 'I', 'read', 'somewhere', 'either', 'do', "n't", 'care', 'for', 'opera', 'do', "n't", 'care', 'for', 'Wagner', 'or', 'do', "n't", 'care', 'about', 'anything', 'except', 'their', 'desire', 'to', 'appear', 'Cultured', '.', 'Either', 'as', 'a', 'representation', 'of', 'Wagner', "'s", 'swan-song', 'or', 'as', 'a', 'movie', 'this', 'strikes', 'me', 'as', 'an', 'unmitigated', 'disaster', 'with', 'a', 'leaden', 'reading', 'of', 'the', 'score', 'matched', 'to', 'a', 'tricksy', 'lugubrious', 'realisation', 'of', 'the', 'text.It', "'s", 'questionable', 'that', 'people', 'with', 'ideas', 'as', 'to', 'what', 'an', 'opera', 'or', 'for', 'that', 'matter', 'a', 'play', 'especially', 'one', 'by', 'Shakespeare', 'is', 'about', 'should', 'be', 'allowed', 'anywhere', 'near', 'a', 'theatre', 'or', 'film', 'studio', 'Syberberg', 'very', 'fashionably', 'but', 'without', 'the', 'smallest', 'justification', 'from', 'Wagner', "'s", 'text', 'decided', 'that', 'Parsifal', 'is', 'about', 'bisexual', 'integration', 'so', 'that', 'the', 'title', 'character', 'in', 'the', 'latter', 'stages', 'transmutes', 'into', 'a', 'kind', 'of', 'beatnik', 'babe', 'though', 'one', 'who', 'continues', 'to', 'sing', 'high', 'tenor', 'few', 'if', 'any', 'of', 'the', 'actors', 'in', 'the', 'film', 'are', 'the', 'singers', 'and', 'we', 'get', 'a', 'double', 'dose', 'of', 'Armin', 'Jordan', 'the', 'conductor', 'who', 'is', 'seen', 'as', 'the', 'face', 'but', 'not', 'heard', 'as', 'the', 'voice', 'of', 'Amfortas', 'and', 'also', 'appears',
    'monstrously', 'in', 'double', 'exposure', 'as', 'a', 'kind', 'of', 'Batonzilla', 'or', 'Conductor', 'Who', 'Ate', 'Monsalvat', 'during', 'the', 'playing', 'of', 'the', 'Good', 'Friday', 'music', 'in', 'which', 'by', 'the', 'way', 'the', 'transcendant', 'loveliness', 'of', 'nature', 'is', 'represented', 'by', 'a', 'scattering', 'of', 'shopworn', 'and', 'flaccid', 'crocuses', 'stuck', 'in', 'ill-laid', 'turf', 'an', 'expedient', 'which', 'baffles', 'me', '.', 'In', 'the', 'theatre', 'we', 'sometimes', 'have', 'to', 'piece', 'out', 'such', 'imperfections', 'with', 'our', 'thoughts', 'but', 'I', 'ca', "n't", 'think', 'why', 'Syberberg', 'could', "n't", 'splice', 'in', 'for', 'Parsifal', 'and', 'Gurnemanz', 'mountain', 'pasture', 'as', 'lush', 'as', 'was', 'provided', 'for', 'Julie', 'Andrews', 'in', 'Sound', 'of', 'Music', 'The', 'sound', 'is', 'hard', 'to', 'endure', 'the', 'high', 'voices', 'and', 'the', 'trumpets', 'in', 'particular', 'possessing', 'an', 'aural', 'glare', 'that', 'adds', 'another', 'sort', 'of', 'fatigue', 'to', 'our', 'impatience', 'with', 'the', 'uninspired', 'conducting', 'and', 'paralytic', 'unfolding', 'of', 'the', 'ritual', '.', 'Someone', 'in', 'another', 'review', 'mentioned', 'the', '1951', 'Bayreuth', 'recording', 'and', 'Knappertsbusch', 'though', 'his', 'tempi', 'are', 'often', 'very', 'slow', 'had', 'what', 'Jordan', 'altogether', 'lacks', 'a', 'sense', 'of', 'pulse', 'a', 'feeling', 'for', 'the', 'ebb', 'and', 'flow', 'of', 'the', 'music', 'and', 'after', 'half', 'a', 'century', 'the', 'orchestral', 'sound', 'in', 'that', 'set', 'in', 'modern', 'pressings', 'is', 'still', 'superior', 'to', 'this', 'film', '.']


"""preprocess text by removing html syntax, unimportant punctuation marks (e.g. ..., --), lowering case, and lemmatizing"""

WNlemma_n = nltk.WordNetLemmatizer()


def text_preprocess(text):
    # remove html syntax
    text1 = BeautifulSoup(text).get_text()
    # get rid of unimportant punctuation marks
    text1 = re.sub(r'([^\d\w\'\s\.\-]+|[-\.]{2,})', ' ', text1)
    # lower case
    text1 = text1.lower()
    # lemmatise
    text1 = WNlemma_n.lemmatize(text1)
    return text1


# print(text_preprocess(labeledTrainData0['review'].iloc[20]))


def pd_text_preprocess(row):
    row['review_preprocessed'] = text_preprocess(row['review'])
    return row


labeledTrainData0 = labeledTrainData0.apply(pd_text_preprocess, axis=1)

testData = testData.apply(pd_text_preprocess, axis=1)
X_test = testData['review_preprocessed']


# %%
