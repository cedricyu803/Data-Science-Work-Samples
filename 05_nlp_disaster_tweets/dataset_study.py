# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:00:00 2021

@author: Cedric Yu
"""

# import matplotlib.pyplot as plt
# import math
# import re


"""
#####################################

# https://www.kaggle.com/c/nlp-getting-started/overview

Competition Description
Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

But, it’s not always clear whether a person’s words are actually announcing a disaster. In this competition, you’re challenged to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t. 

#####################################

# Metric
Submissions are evaluated using [F1 between the predicted and expected answers].

Submission Instructions
For each ID in the test set, you must predict 1 if the tweet is describing a real disaster, and 0 otherwise. The file should contain a header and have the following format:

id,target
0,0
2,0
3,1
9,0
11,0

#####################################

# Dataset

Files

# train.csv - the training set
# test.csv - the test set
# sample_submission.csv - a sample submission file in the correct format

Columns

# id - a unique identifier for each tweet
# text - the text of the tweet
# location - the location the tweet was sent from (may be blank)
# keyword - a particular keyword from the tweet (may be blank)
# target - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)


"""

# %% This file

"""
exploratory data analysis

"""


# %% Preamble

# Make the output look better
from nltk.corpus import stopwords
from geotext import GeoText
from bs4 import BeautifulSoup
import pycountry
import os
import seaborn as sns
import re
import matplotlib.pyplot as plt
import numpy as np
import nltk
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
# default='warn' # ignores warning about dropping columns inplace
pd.options.mode.chained_assignment = None

os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\9_nlp_disaster_tweets')

# %% load datasets

train = pd.read_csv("train.csv", header=[0])
test = pd.read_csv("test.csv", header=[0])

train0 = train.copy()
test0 = test.copy()

train0['target'].mean()
# 0.4296597924602653

train0.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 7613 entries, 0 to 7612
# Data columns (total 5 columns):
#  #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   id        7613 non-null   int64
#  1   keyword   7552 non-null   object
#  2   location  5080 non-null   object
#  3   text      7613 non-null   object
#  4   target    7613 non-null   int64
# dtypes: int64(2), object(3)

train0.isnull().sum()
# id             0
# keyword       61
# location    2533
# text           0
# target         0
# text_len       0
# dtype: int64
"""# a lot of nan in 'location'
# some nan in 'keyword'"""

test0.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 3263 entries, 0 to 3262
# Data columns (total 4 columns):
#  #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   id        3263 non-null   int64
#  1   keyword   3237 non-null   object
#  2   location  2158 non-null   object
#  3   text      3263 non-null   object
# dtypes: int64(1), object(3)
# memory usage: 102.1+ KB

test0.isnull().sum()
# id             0
# keyword       26
# location    1105
# text           0
# dtype: int64
"""# a lot of nan in 'location'"""


train0.head()
#    id keyword location  \
# 0   1     NaN      NaN
# 1   4     NaN      NaN
# 2   5     NaN      NaN
# 3   6     NaN      NaN
# 4   7     NaN      NaN

#                                                                                                                                     text  \
# 0                                                                  Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all
# 1                                                                                                 Forest fire near La Ronge Sask. Canada
# 2  All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected
# 3                                                                      13,000 people receive #wildfires evacuation orders in California
# 4                                               Just got sent this photo from Ruby #Alaska as smoke from #wildfires pours into a school

#    target
# 0       1
# 1       1
# 2       1
# 3       1
# 4       1

# experimenting with regex on #, @ and urls

for i in range(100):
    example = re.findall(r'(://[\s]*)', train0['text'].iloc[i])
    example = list(filter(None, example))
    if len(example) > 0:
        print(i)
        print(example)

train0['text'].iloc[36]
# "@PhDSquares #mufc they've built so much hype around new acquisitions but I doubt they will set the EPL ablaze this season."

train0['text'].iloc[31]
# '@bbcmtd Wholesale Markets ablaze http://t.co/lHYXEOHY6C'

train0['text'].iloc[57]
# 'Set our hearts ablaze and every city was a gift And every skyline was like a kiss upon the lips @\x89Û_ https://t.co/cYoMPZ1A0Z'

train0['text'].iloc[83]
# Out[205]: "#TruckCrash Overturns On #FortWorth Interstate http://t.co/Rs22LJ4qFp Click here if you've been in a crash&gt;http://t.co/Ld0unIYw4k"

""" 
!!! things appearing in a Tweet: hashtags, @, URLs (http: and https:)
Mojibake needs processing
"""

# %% character length before any pre-processing

"""character length"""
# twitter has 280 character limit


def char_count(row):
    row['char_count'] = len(row['text'])
    return row


train0 = train0.apply(char_count, axis=1)
train0['char_count'].describe()
# count    7613.000000
# mean      101.037436
# std        33.781325
# min         7.000000
# 25%        78.000000
# 50%       107.000000
# 75%       133.000000
# max       157.000000
# Name: char_count, dtype: float64

test0 = test0.apply(char_count, axis=1)


fig = plt.figure(dpi=150)
train0['char_count'].hist(grid=False, density=True,
                          ax=plt.gca(), color='skyblue')
train0['char_count'].plot.kde(grid=False, ax=plt.gca(), color='black')
ax = plt.gca()
ax.set_xlabel('Character length')
ax.set_ylabel(None)
ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0.1, 200)
# plt.savefig('plots/char_len/train', dpi = 150)

fig = plt.figure(dpi=150)
test0['char_count'].hist(grid=False, density=True,
                         ax=plt.gca(), color='tomato')
test0['char_count'].plot.kde(grid=False, ax=plt.gca(), color='black')
ax = plt.gca()
ax.set_xlabel('Character length')
ax.set_ylabel(None)
ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0.1, 200)
# plt.savefig('plots/char_len/test', dpi = 150)


fig = plt.figure(dpi=150)
train0['char_count'].hist(grid=False, density=True,
                          ax=plt.gca(), color='skyblue', alpha=0.5, align='mid')
plt.yscale('log')
# train0['char_count'].plot.kde(grid = False, ax = plt.gca(), color = 'deepskyblue')
test0['char_count'].hist(grid=False, density=True,
                         ax=plt.gca(), color='tomato', alpha=0.5, align='mid')
# test0['char_count'].plot.kde(grid = False, ax = plt.gca(), color = 'orangered')
ax = plt.gca()
ax.set_xlabel('Character length')
ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(0, 160)
# plt.savefig('plots/char_len/train_test_logscale', dpi = 150)

"""
!!! train and test have almost the same distribution
"""

# %% keyword

train0[train0['keyword'].isnull() != True].head()
#     id keyword                       location  \
# 31  48  ablaze                     Birmingham
# 32  49  ablaze  Est. September 2012 - Bristol
# 33  50  ablaze                         AFRICA
# 34  52  ablaze               Philadelphia, PA
# 35  53  ablaze                     London, UK

#                                                                                   text  \
# 31                             @bbcmtd Wholesale Markets ablaze http://t.co/lHYXEOHY6C
# 32                 We always try to bring the heavy. #metal #RT http://t.co/YAo1e0xngw
# 33  #AFRICANBAZE: Breaking news:Nigeria flag set ablaze in Aba. http://t.co/2nndBGwyEi
# 34                                                  Crying out for more! Set me ablaze
# 35        On plus side LOOK AT THE SKY LAST NIGHT IT WAS ABLAZE http://t.co/qqsmshaJ3N

#     target  text_len
# 31       1        55
# 32       0        67
# 33       1        82
# 34       0        34
# 35       0        76


train0['keyword'].unique()
# array([nan, 'ablaze', 'accident', 'aftershock', 'airplane%20accident',
#        'ambulance', 'annihilated', 'annihilation', 'apocalypse',
#        'armageddon', 'army', 'arson', 'arsonist', 'attack', 'attacked',
#        'avalanche', 'battle', 'bioterror', 'bioterrorism', 'blaze',
#        'blazing', 'bleeding', 'blew%20up', 'blight', 'blizzard', 'blood',
#        'bloody', 'blown%20up', 'body%20bag', 'body%20bagging',
#        'body%20bags', 'bomb', 'bombed', 'bombing', 'bridge%20collapse',
#        'buildings%20burning', 'buildings%20on%20fire', 'burned',
#        'burning', 'burning%20buildings', 'bush%20fires', 'casualties',
#        'casualty', 'catastrophe', 'catastrophic', 'chemical%20emergency',
#        'cliff%20fall', 'collapse', 'collapsed', 'collide', 'collided',
#        'collision', 'crash', 'crashed', 'crush', 'crushed', 'curfew',
#        'cyclone', 'damage', 'danger', 'dead', 'death', 'deaths', 'debris',
#        'deluge', 'deluged', 'demolish', 'demolished', 'demolition',
#        'derail', 'derailed', 'derailment', 'desolate', 'desolation',
#        'destroy', 'destroyed', 'destruction', 'detonate', 'detonation',
#        'devastated', 'devastation', 'disaster', 'displaced', 'drought',
#        'drown', 'drowned', 'drowning', 'dust%20storm', 'earthquake',
#        'electrocute', 'electrocuted', 'emergency', 'emergency%20plan',
#        'emergency%20services', 'engulfed', 'epicentre', 'evacuate',
#        'evacuated', 'evacuation', 'explode', 'exploded', 'explosion',
#        'eyewitness', 'famine', 'fatal', 'fatalities', 'fatality', 'fear',
#        'fire', 'fire%20truck', 'first%20responders', 'flames',
#        'flattened', 'flood', 'flooding', 'floods', 'forest%20fire',
#        'forest%20fires', 'hail', 'hailstorm', 'harm', 'hazard',
#        'hazardous', 'heat%20wave', 'hellfire', 'hijack', 'hijacker',
#        'hijacking', 'hostage', 'hostages', 'hurricane', 'injured',
#        'injuries', 'injury', 'inundated', 'inundation', 'landslide',
#        'lava', 'lightning', 'loud%20bang', 'mass%20murder',
#        'mass%20murderer', 'massacre', 'mayhem', 'meltdown', 'military',
#        'mudslide', 'natural%20disaster', 'nuclear%20disaster',
#        'nuclear%20reactor', 'obliterate', 'obliterated', 'obliteration',
#        'oil%20spill', 'outbreak', 'pandemonium', 'panic', 'panicking',
#        'police', 'quarantine', 'quarantined', 'radiation%20emergency',
#        'rainstorm', 'razed', 'refugees', 'rescue', 'rescued', 'rescuers',
#        'riot', 'rioting', 'rubble', 'ruin', 'sandstorm', 'screamed',
#        'screaming', 'screams', 'seismic', 'sinkhole', 'sinking', 'siren',
#        'sirens', 'smoke', 'snowstorm', 'storm', 'stretcher',
#        'structural%20failure', 'suicide%20bomb', 'suicide%20bomber',
#        'suicide%20bombing', 'sunk', 'survive', 'survived', 'survivors',
#        'terrorism', 'terrorist', 'threat', 'thunder', 'thunderstorm',
#        'tornado', 'tragedy', 'trapped', 'trauma', 'traumatised',
#        'trouble', 'tsunami', 'twister', 'typhoon', 'upheaval',
#        'violent%20storm', 'volcano', 'war%20zone', 'weapon', 'weapons',
#        'whirlwind', 'wild%20fires', 'wildfire', 'windstorm', 'wounded',
#        'wounds', 'wreck', 'wreckage', 'wrecked'], dtype=object)


train0[['keyword', 'text']].iloc[299]
# keyword                                                                                                           apocalypse
# text       The latest from @BryanSinger reveals #Storm is a queen in #Apocalypse @RuPaul @AlexShipppp http://t.co/oQw8Jx6rTs


keyword_target = train0.groupby('keyword')['target'].agg(np.nanmean)
keyword_target = keyword_target.sort_values(ascending=True)
keyword_target_cut = keyword_target.iloc[-30:]
keyword_target_cut2 = keyword_target.iloc[:30]


fig = plt.figure(dpi=150)
keyword_target_cut.plot.barh(grid=False, ax=plt.gca(), color='skyblue')
ax = plt.gca()
ax.set_xlabel('Target mean')
ax.set_ylabel('keyword')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xlim(0.6, 1)
plt.tight_layout()
# plt.savefig('plots/keyword_train', dpi = 150)

fig = plt.figure(dpi=150)
keyword_target_cut2.plot.barh(grid=False, ax=plt.gca(), color='skyblue')
ax = plt.gca()
ax.set_xlabel('Target mean')
ax.set_ylabel('keyword')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xlim(0, .14)
plt.tight_layout()
# plt.savefig('plots/keyword_train2', dpi = 150)

"""
!!! can be highly indicative of a disaster, but has hyperbolae, e.g. 'apocalypse'. Use target encoding for each keyword
"""

# %% location

len(set(train0['location'].unique()))
# 3342

train_locations = set(train0['location'].unique())

# very messy
# {....
# 'Long Island',
# 'My mind is my world',
# 'Detroit Tigers Dugout',
# 'Victoria, BC  Canada',
# 'Wolverhampton/Brum/Jersey',
# '?? ?+254? ? \\??å¡_??å¡_???å¡_?/??',
# 'Central Illinois',
# 'Fruit Bowl',
# 'Glenview to Knoxville ',
# 'Johannesburg ',
# 'Moore, OK',
# 'South West, England',
# 'Brisbane.',
# 'Illumination ',
# 'Asia',
# 'Intramuros, Manila',
# 'The Jewfnited State',
# 'Coolidge, AZ',
# 'Vancouver BC',
# 'Abuja,Nigeria',
# 'Escondido, CA',
# 'im definitely taller than you.',
# 'Gaborone, Botswana',
# 'Los Angeles, London, Kent',
# 'Third rock from the Sun',
# 'Giddy, Greenland',
# "Viterbo BFA Acting '18",
# 'Made Here In Detroit ',
# 'Helsinki',
# " 45å¡ 5'12.53N   14å¡ 7'24.93E",
# 'Surrey & Manchester',
# 'Rock Springs, WY',
# 'snapchat~ maddzz_babby ',
# 'Chicago, Illinois',
# 'Dubai, UAE',
# 'Everett, WA',
# 'chicago',
# 'Football Field',
# 'Subconscious LA',
# 'PG County, MD',
# 'ill yorker',
# 'White Plains, NY',
# 'houstn',
# 'HTX',
# ...}

text0 = "United States (New York), United Kingdom (London)"
for country in pycountry.countries:
    if country.name in text0:
        print(country.name)


"""
!!! too many nan, drop.
"""

# %% remove mojibake


def no_mojibake(row):
    row['text_no_mojibake'] = BeautifulSoup(row['text']).get_text().strip()
    return row


train0 = train0.apply(no_mojibake, axis=1)
test0 = test0.apply(no_mojibake, axis=1)


"""
!!! work with 'text_no_mojibake' from now on
"""


# %% character count

def char_count_no_moj(row):
    row['char_count'] = len(row['text_no_mojibake'])
    return row


train0 = train0.apply(char_count_no_moj, axis=1)
test0 = test0.apply(char_count_no_moj, axis=1)

"""
!!! use this char count
"""


# %% punctuation and capital letter character ratios

"""
!!! # record the number of words, characters, capital letters and punctuation marks
"""


def punc_cap_ratio(row):
    row['punc_ratio'] = len(''.join(re.findall(
        r'[\.\?!\"#$%\&\'\(\)\*\+,\-/:;=@\[\]\^_`\{\|\}~\\]+', row['text_no_mojibake']))) / len(row['text_no_mojibake'])
    row['cap_ratio'] = len(''.join(re.findall(
        r'[A-Z]', row['text_no_mojibake']))) / len(row['text_no_mojibake'])
    return row


train0 = train0.apply(punc_cap_ratio, axis=1)
test0 = test0.apply(punc_cap_ratio, axis=1)


fig = plt.figure(dpi=150)
train0['punc_ratio'].hist(grid=False, density=True,
                          ax=plt.gca(), color='skyblue', alpha=0.5)
# train0['punc_ratio'].plot.kde(grid = False, ax = plt.gca(), color = 'deepskyblue')
test0['punc_ratio'].hist(grid=False, density=True,
                         ax=plt.gca(), color='tomato', alpha=0.5)
# test0['punc_ratio'].plot.kde(grid = False, ax = plt.gca(), color = 'orangered')
ax = plt.gca()
plt.yscale('log')
ax.set_xlabel('Ratio of punctuation marks to character length')
ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0, 0.5)
# plt.savefig('plots/punc/train_test_logscale', dpi = 150)


fig = plt.figure(dpi=150)
train0['cap_ratio'].hist(grid=False, density=True,
                         ax=plt.gca(), color='skyblue', alpha=0.5)
# train0['cap_ratio'].plot.kde(grid = False, ax = plt.gca(), color = 'deepskyblue')
test0['cap_ratio'].hist(grid=False, density=True,
                        ax=plt.gca(), color='tomato', alpha=0.5)
# test0['cap_ratio'].plot.kde(grid = False, ax = plt.gca(), color = 'orangered')
ax = plt.gca()
plt.yscale('log')
ax.set_xlabel('Ratio of capital letters to character length')
ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(0, 1)
# plt.savefig('plots/cap_letters/train_test_logscale', dpi = 150)


# %% hashtags, mentions @ and URLs

train0['text'].iloc[0]
example1 = re.findall(
    r'#([^#\s]*)', 'Our Deeds are the ##Reason of this #earthquake May ALLAH Forgive us all')
example1 = list(filter(None, example1))
# ['Reason', 'earthquake']

train0['text'].iloc[1]
example2 = re.findall(r'#([^#\s]*)', 'Forest fire near La Ronge Sask. Canada')
example2 = list(filter(None, example2))
# []

# re.findall(r"\'{1}([^\']+)\'{1}", "['Reason', 'earthquake']")

hashtags_lists = []


def get_hashtags(row):
    hashtags_list = re.findall(r'#(\w+)', row['text_no_mojibake'])
    hashtags_list = list(filter(None, hashtags_list))
    row['hashtags'] = hashtags_list
    row['hashtag_num'] = len(hashtags_list)
    for hashtag in hashtags_list:
        hashtags_lists.append(hashtag)
    return row


def get_hashtags_test(row):
    hashtags_list = re.findall(r'#(\w+)', row['text_no_mojibake'])
    hashtags_list = list(filter(None, hashtags_list))
    row['hashtags'] = hashtags_list
    row['hashtag_num'] = len(hashtags_list)
    return row


train0 = train0.apply(get_hashtags, axis=1)
# len(set(hashtags_lists))
# Out[201]: 2141
test0 = test0.apply(get_hashtags_test, axis=1)

# '8/6/2015@2:09 PM: TRAFFIC ACCIDENT NO INJURY at 2781 WILLIS FOREMAN RD http://t.co/VCkIT6EDEv'
# "\x89ÛÏ@LeoBlakeCarter: This dog thinks he's an ambulance ?????? http://t.co/MG1lpGr0RM\x89Û\x9d@natasha_rideout"

at_lists = []


def get_at(row):
    # if there is no space in front of @, it may actually show up in a url
    at_list = re.findall(r'@(\w+)', row['text_no_mojibake'])
    at_list = list(filter(None, at_list))
    row['at'] = at_list
    row['at_num'] = len(at_list)
    for at in at_list:
        at_lists.append(at)
    return row


def get_at_test(row):
    at_list = re.findall(r'@(\w+)', row['text_no_mojibake'])
    at_list = list(filter(None, at_list))
    row['at'] = at_list
    row['at_num'] = len(at_list)
    return row


train0 = train0.apply(get_at, axis=1)
# len(set(at_lists))
# Out[201]: 2326
test0 = test0.apply(get_at_test, axis=1)


def get_num_url(row):
    url_list = re.findall(r'(https?://[\S]*)', row['text_no_mojibake'])
    url_list = list(filter(None, url_list))
    row['url'] = url_list
    row['url_num'] = len(url_list)
    return row


train0 = train0.apply(get_num_url, axis=1)
test0 = test0.apply(get_num_url, axis=1)


fig = plt.figure(dpi=150)
train0['hashtag_num'].hist(grid=False, density=True,
                           ax=plt.gca(), color='skyblue', alpha=0.5, align='mid')
# train0['hashtag_num'].plot.kde(grid = False, ax = plt.gca(), color = 'deepskyblue')
test0['hashtag_num'].hist(grid=False, density=True,
                          ax=plt.gca(), color='tomato', alpha=0.5, align='mid')
# test0['hashtag_num'].plot.kde(grid = False, ax = plt.gca(), color = 'orangered')
ax = plt.gca()
plt.yscale('log')
ax.set_xlabel('Number of hashtags in each Tweet')
ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0, 12)
# plt.savefig('plots/hashtag_mention_url/hashtag_num_train_test_logscale', dpi = 150)


fig = plt.figure(dpi=150)
train0['at_num'].hist(grid=False, bins=9, density=True,
                      ax=plt.gca(), color='skyblue', alpha=0.5, align='mid')
# train0['at_num'].plot.kde(grid = False, ax = plt.gca(), color = 'deepskyblue')
test0['at_num'].hist(grid=False, bins=9, density=True,
                     ax=plt.gca(), color='tomato', alpha=0.5, align='mid')
# test0['at_num'].plot.kde(grid = False, ax = plt.gca(), color = 'orangered')
ax = plt.gca()
plt.yscale('log')
ax.set_xlabel('Number of mentions (@) in each Tweet')
ax.set_ylabel(None)
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0, 8)
# plt.savefig('plots/hashtag_mention_url/mention_num_train_test_logscale', dpi = 150)


fig = plt.figure(dpi=150)
train0['url_num'].hist(grid=False, bins=4, density=True,
                       ax=plt.gca(), color='skyblue', alpha=0.5, align='mid')
# train0['url_num'].plot.kde(grid = False, ax = plt.gca(), color = 'deepskyblue')
test0['url_num'].hist(grid=False, bins=4, density=True,
                      ax=plt.gca(), color='tomato', alpha=0.5, align='mid')
# test0['url_num'].plot.kde(grid = False, ax = plt.gca(), color = 'orangered')
ax = plt.gca()
ax.set_xlabel('Number of URLs in each Tweet')
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(0, 4)
# plt.savefig('plots/hashtag_mention_url/url_num_train_test', dpi = 150)


"""
target mean grouped by number of #, @ and urls
"""

train0_by_hashtag_num = train0[['hashtag_num', 'target']].groupby(
    ['hashtag_num']).agg(np.mean)

train0_by_at_num = train0[['at_num', 'target']
                          ].groupby(['at_num']).agg(np.mean)

train0_by_url_num = train0[['url_num', 'target']
                           ].groupby(['url_num']).agg(np.mean)


fig = plt.figure(dpi=150)
train0_by_hashtag_num.plot(kind='bar', xlabel='hashtag_num', ax=plt.gca(
), color='skyblue', alpha=0.5, align='center')
ax = plt.gca()
ax.set_xlabel('Number of hashtags in each Tweet')
# ax.set_xticks([0,1,2,3,4])
# ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_legend().remove()
# plt.xlim(0,4)
# plt.savefig('plots/hashtag_mention_url/hashtag_num_train_target', dpi = 150)


fig = plt.figure(dpi=150)
train0_by_at_num.plot(kind='bar', xlabel='at_num',
                      ax=plt.gca(), color='skyblue', alpha=0.5, align='center')
ax = plt.gca()
ax.set_xlabel('Number of mentions in each Tweet')
# ax.set_xticks([0,1,2,3,4])
# ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_legend().remove()
# plt.xlim(0,4)
# plt.savefig('plots/hashtag_mention_url/at_num_train_target', dpi = 150)


fig = plt.figure(dpi=150)
train0_by_url_num.plot(kind='bar', xlabel='url_num',
                       ax=plt.gca(), color='skyblue', alpha=0.5, align='center')
ax = plt.gca()
ax.set_xlabel('Number of URLs in each Tweet')
# ax.set_xticks([0,1,2,3,4])
# ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_legend().remove()
# plt.xlim(0,4)
# plt.savefig('plots/hashtag_mention_url/url_num_train_target', dpi = 150)


# re.findall(r"\'{1}([^\']+)\'{1}", "['Reason', 'earthquake']")


"""
!!! extract number of hashtags and at, 
drop the #, @ and url when pre-process text
Too many unique hashtags and at to process one-by-one
numbers not very correlated with target; frequency encode
"""


# %% sentence count


train0 = train0.apply(get_num_url, axis=1)
test0 = test0.apply(get_num_url, axis=1)


def sent_count(row):
    row['sentence_count'] = len(nltk.sent_tokenize(row['text_no_mojibake']))
    return row


train0 = train0.apply(sent_count, axis=1)
test0 = test0.apply(sent_count, axis=1)


fig = plt.figure(dpi=150)
train0['sentence_count'].hist(grid=False, bins=10, density=True, ax=plt.gca(
), color='skyblue', alpha=0.5, align='mid')
test0['sentence_count'].hist(grid=False, bins=10, density=True, ax=plt.gca(
), color='tomato', alpha=0.5, align='mid')
ax = plt.gca()
ax.set_xlabel('Number of sentences in each Tweet')
plt.yscale('log')
# ax.set_xticks([0,1,2,3,4])
ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(1, 15)
# plt.savefig('plots/sent_count/train_test', dpi = 150)


"""
target mean grouped by number of sentences
"""

train0_by_sent_count = train0[['sentence_count', 'target']].groupby(
    ['sentence_count']).agg(np.mean)


fig = plt.figure(dpi=150)
train0_by_sent_count.plot(kind='bar', xlabel='sentence_count', ax=plt.gca(
), color='skyblue', alpha=0.5, align='center')
ax = plt.gca()
ax.set_xlabel('Number of sentences in each Tweet')
# ax.set_xticks([0,1,2,3,4])
# ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_legend().remove()
# plt.xlim(0,4)
# plt.savefig('plots/sent_count/sent_count_train_target', dpi = 150)

"""
!!! not correlated with target; frequency encode
"""


# %% stopword numbers

"""
text_no_mojibake VS text_processed: distribution looks basically the same
"""

stops = set(stopwords.words('english'))


def stop_word_num(row):
    tokens = nltk.word_tokenize(row['text_no_mojibake'])
    stop_words = len([token for token in tokens if token in stops])
    row['stopword_num'] = stop_words
    return row


train0 = train0.apply(stop_word_num, axis=1)
test0 = test0.apply(stop_word_num, axis=1)


fig = plt.figure(dpi=150)
train0['stopword_num'].hist(
    grid=False, density=True, ax=plt.gca(), color='skyblue', alpha=0.5, align='mid')
test0['stopword_num'].hist(grid=False, density=True,
                           ax=plt.gca(), color='tomato', alpha=0.5, align='mid')
ax = plt.gca()
ax.set_xlabel('Number of stopwords in each Tweet')
ax.set_xticks(np.arange(0, 19, 4))
ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(0, 16)
# plt.savefig('plots/stopword_num/stopword_num_train_test', dpi = 150)

"""
target mean grouped by number of stopwords
"""

train0_by_stopword_num = train0[['stopword_num', 'target']].groupby(
    ['stopword_num']).agg(np.mean)


fig = plt.figure(dpi=150)
train0_by_stopword_num.plot(kind='bar', xlabel='stopword_num', ax=plt.gca(
), color='skyblue', alpha=0.5, align='center')
ax = plt.gca()
ax.set_xlabel('Number of stopwords in each Tweet')
# ax.set_xticks([0,1,2,3,4])
# ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_legend().remove()
# plt.xlim(0,4)
# plt.savefig('plots/stopword_num/stopword_num_train_target', dpi = 150)

"""
!!! not correlated with target; frequency encode
"""

# %% geodata


def country_mentions_num(row):
    if len(list(zip(*GeoText(row['text_no_mojibake']).country_mentions.items()))) > 0:
        row['country_mention_num'] = np.array(
            list(zip(*GeoText(row['text_no_mojibake']).country_mentions.items()))[1]).sum()
    else:
        row['country_mention_num'] = 0
    return row


train0 = train0.apply(country_mentions_num, axis=1)
test0 = test0.apply(country_mentions_num, axis=1)

# train0['country_mention_num'].value_counts()
# Out[13]:
# 0    6306
# 1    1053
# 2     230
# 3      20
# 5       2
# 4       1
# 6       1
# Name: country_mention_num, dtype: int64


fig = plt.figure(dpi=150)
train0['country_mention_num'].hist(grid=False, bins=6, density=True, ax=plt.gca(
), color='skyblue', alpha=0.5, align='mid')
test0['country_mention_num'].hist(grid=False, bins=6, density=True, ax=plt.gca(
), color='tomato', alpha=0.5, align='mid')
ax = plt.gca()
plt.yscale('log')
ax.set_xlabel('Number of country mentions in each Tweet')
# ax.set_xticks(np.arange(0,19,4))
ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(0, 6)
# plt.savefig('plots/country_mentions/country_mentions_train_test', dpi = 150)

"""
target mean grouped by number of country mentions
"""

train0_by_country_mention_num = train0[['country_mention_num', 'target']].groupby(
    ['country_mention_num']).agg(np.mean)


fig = plt.figure(dpi=150)
train0_by_country_mention_num.plot(kind='bar', xlabel='country_mention_num', ax=plt.gca(
), color='skyblue', alpha=0.5, align='center')
ax = plt.gca()
ax.set_xlabel('Number of country mentions in each Tweet')
# ax.set_xticks([0,1,2,3,4])
# ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_legend().remove()
# plt.xlim(0,4)
# plt.savefig('plots/country_mentions/country_mentions_train_target', dpi = 150)

"""
!!! frequency encode
"""


# %% experimenting with pre-processing

"""
# we will use GloVe Twitter embeddings; see script on the website

# remove HTML syntax
# strip front and end whitespaces
# expand abbreviation n't to not and 's to  s
# record the ratios of capital letters and punctuation marks to total characters
# replace url by '<url>' token
# replace @mention by '<user>' token
# replace # by <hashtag> token, and split the hashtag_body by capital letters unless it is all cap
# represent numbers by '<number>' token
# replace repeated punctuation marks by punctuation mark + <repeat> token
# replace punctuation marks (except ?!.) by <punc> token
# remove extra whitespaces
# (we do not handle emojis) in this exercise
# (remove stop words? Do not!)
# for nltk vectorisers:
# lower case
# lemmatize
for LSTM:
# lower case
# pad sequences by hand to max_len=? (max char length is 280)
for transformer: 
# not much else to do

"""


"""
we loosely follow 
https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""


def text_preprocess(text):
    # remove html syntax and strip front and end whitespace (already did)
    # text1 = BeautifulSoup(text).get_text()

    # expand abbreviation n't to not and 's to  s
    text1 = re.sub(r"n't", ' not', text)
    text1 = re.sub(r"'s", ' s', text1)

    # replace url by '<url>' token
    text1 = re.sub(r'(https?://[\S]*)', '<url>', text1)
    # replace @mention by '<user>' token
    text1 = re.sub(r'@\w+', "<user>", text1)

    # replace # by <hashtag> token, and split the hashtag_body by capital letters unless it is all cap
    hash_iter = list(filter(None, re.findall(r'#(\w+)', text1)))
    if len(hash_iter) > 0:
        for item in hash_iter:
            if item.isupper() == True:
                hash_words = item + ' <allcaps>'
            else:
                hash_words = ' '.join(
                    list(filter(None, re.split(r'([A-Z]?[a-z]*)', item))))
            text1 = re.sub(item, hash_words, text1)
            text1 = re.sub(r'#', '<hashtag> ', text1)

    # represent numbers by '<number>' token
    text1 = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', ' <number> ', text1)

    # replace repeated punctuation marks (!?. only) by punctuation mark + <repeat> token
    text1 = re.sub(r'([!?\.]){2,}', r'\1' + ' <repeat>', text1)
    # add spaces before and after (!?.)
    text1 = re.sub(r'([!?\.])', r' \1 ', text1)

    # replace punctuation marks (except ?!.) by <punc> token
    text1 = re.sub(
        r'[\"#$%\&\'\(\)\*\+,\-/:;=@\[\]\^_`\{\|\}~\\]+', ' <punc> ', text1)
    text1 = re.sub(r'(<[^A-Za-z0-9_-]+|[^A-Za-z0-9_-]+>)', ' <punc> ', text1)

    # remove extra whitespaces
    text1 = text1.strip()
    text1 = re.sub('\s+', ' ', text1)

    # lemmatise
    # text1 = WNlemma_n.lemmatize(text1)

    # lower case
    text1 = text1.lower()

    return text1


def pd_text_preprocess(row):
    row['text_processed'] = text_preprocess(row['text_no_mojibake'])
    return row


WNlemma_n = nltk.WordNetLemmatizer()


def nltk_lemmatize(text):
    # text1 = nltk.word_tokenize(text)
    text1 = WNlemma_n.lemmatize(text)
    return text1


def pd_nltk_lemmatize(row):
    row['text_processed'] = nltk_lemmatize(row['text_processed'])
    return row


def pad_text(text, max_len=280):
    # pad to max_len by '-1 empty'
    text1 = text
    if len(text) < max_len:
        text1 += ['-1 empty' for i in range(max_len - len(text))]
    return text1


def pd_pad_text(row):
    row['text_processed'] = pad_text(row['text_processed'])
    return row


train0 = train0.apply(pd_text_preprocess, axis=1)
test0 = test0.apply(pd_text_preprocess, axis=1)


# %% load embedding matrix pre-trained using glove

# https://nlp.stanford.edu/projects/glove/
# Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors)
# read in the glove file. Each line in the text file is a string containing a word followed by the embedding vector, all separated by a whitespace
# word_to_vec_map is a dict of words to their embedding vectors
# (~1.2M words, with the valid indices starting from 1 to ~1.2M

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


# index is valued  from 1 to ~1.2M
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(
    'glove.twitter.27B/glove.twitter.27B.100d.txt')
# contains <hashtag>, <user>, <url>, <repeat>, <number>


# %% experimenting with feature extractions

"""
# number of characters
# punctuation and capital letter ratio
# sentence count: frequency encode
# number of stopwords: frequency encode
# number of #, @, urls: frequency encode
# keyword: target encode
# country mentions with GeoText(text).country_mentions: frequency encode

hashtag and mentions lists for each Tweet: also pre-process and feed to LSTM?
"""


def punc_cap_ratio(row):
    row['punc_ratio'] = len(''.join(re.findall(
        r'[\.\?!\"#$%\&\'\(\)\*\+,\-/:;=@\[\]\^_`\{\|\}~\\]+', row['text_no_mojibake']))) / len(row['text_no_mojibake'])
    row['cap_ratio'] = len(''.join(re.findall(
        r'[A-Z]', row['text_no_mojibake']))) / len(row['text_no_mojibake'])
    return row


train0 = train0.apply(punc_cap_ratio, axis=1)
test0 = test0.apply(punc_cap_ratio, axis=1)


hashtags_lists = []


def get_hashtags(row):
    hashtags_list = re.findall(r'#(\w+)', row['text_no_mojibake'])
    hashtags_list = list(filter(None, hashtags_list))
    row['hashtags'] = hashtags_list
    row['hashtag_num'] = len(hashtags_list)
    for hashtag in hashtags_list:
        hashtags_lists.append(hashtag)
    return row


def get_hashtags_test(row):
    hashtags_list = re.findall(r'#(\w+)', row['text_no_mojibake'])
    hashtags_list = list(filter(None, hashtags_list))
    row['hashtags'] = hashtags_list
    row['hashtag_num'] = len(hashtags_list)
    return row


train0 = train0.apply(get_hashtags, axis=1)
# len(set(hashtags_lists))
# Out[201]: 2141
test0 = test0.apply(get_hashtags_test, axis=1)
hashtags_lists = list(set(hashtags_lists))


# re.findall(r"\'{1}([^\']+)\'{1}", "['Reason', 'earthquake']")


at_lists = []


def get_at(row):
    # if there is no space in front of @, it may actually show up in a url
    at_list = re.findall(r'@(\w+)', row['text_no_mojibake'])
    at_list = list(filter(None, at_list))
    row['at'] = at_list
    row['at_num'] = len(at_list)
    for at in at_list:
        at_lists.append(at)
    return row


def get_at_test(row):
    at_list = re.findall(r'@(\w+)', row['text_no_mojibake'])
    at_list = list(filter(None, at_list))
    row['at'] = at_list
    row['at_num'] = len(at_list)
    return row


train0 = train0.apply(get_at, axis=1)
# len(set(at_lists))
# Out[201]: 2326
test0 = test0.apply(get_at_test, axis=1)

at_lists = list(set(at_lists))


def get_num_url(row):
    url_list = re.findall(r'(https?://[\S]*)', row['text_no_mojibake'])
    url_list = list(filter(None, url_list))
    row['url'] = url_list
    row['url_num'] = len(url_list)
    return row


train0 = train0.apply(get_num_url, axis=1)
test0 = test0.apply(get_num_url, axis=1)


def sent_count(row):
    row['sentence_count'] = len(nltk.sent_tokenize(row['text_no_mojibake']))
    return row


train0 = train0.apply(sent_count, axis=1)
test0 = test0.apply(sent_count, axis=1)


stops = set(stopwords.words('english'))


def stop_word_num(row):
    tokens = nltk.word_tokenize(row['text_processed'])
    stop_words = len([token for token in tokens if token in stops])
    row['stopword_num'] = stop_words
    return row


train0 = train0.apply(stop_word_num, axis=1)
test0 = test0.apply(stop_word_num, axis=1)


def country_mentions_num(row):
    if len(list(zip(*GeoText(row['text_no_mojibake']).country_mentions.items()))) > 0:
        row['country_mention_num'] = np.array(
            list(zip(*GeoText(row['text_no_mojibake']).country_mentions.items()))[1]).sum()
    else:
        row['country_mention_num'] = 0
    return row


train0 = train0.apply(country_mentions_num, axis=1)
test0 = test0.apply(country_mentions_num, axis=1)
