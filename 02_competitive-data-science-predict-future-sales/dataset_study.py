# -*- coding: utf-8 -*-
"""
Created on Sun Aug 01 11:00:00 2021

@author: Cedric Yu
"""

# import matplotlib.pyplot as plt
# import math
# import re


"""
#####################################

# https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview

# This challenge serves as final project for the "How to win a data science competition" Coursera course.

# In this competition you will work with a challenging time-series dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms - 1C Company. 

# We are asking you to predict total sales (count) <<for every product and store>> in the next <<month>>. By solving this competition you will be able to apply and enhance your data science skills.

#####################################

#!!! Submissions are evaluated by <<root mean squared error (RMSE)>>. 
#!!! <<True target values are clipped into [0,20] range>>

# Submission File

#!!! For each id in the test set, you must predict a <<total number of sales>>. The file should contain a header and have the following format:

# ID,item_cnt_month
# 0,0.5
# 1,0.5
# 2,0.5
# 3,0.5
# etc.

#####################################

# File descriptions
# sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
# test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.
# sample_submission.csv - a sample submission file in the correct format.
# items.csv - supplemental information about the items/products.
# item_categories.csv  - supplemental information about the items categories.
# shops.csv- supplemental information about the shops.

#####################################

# Data fields
# ID - an Id that represents a (Shop, Item) tuple within the <test set>
# shop_id - unique identifier of a shop
# item_id - unique identifier of a product
# item_category_id - unique identifier of item category
# item_cnt_day - number of products sold. You are predicting a monthly amount of this measure
# item_price - current price of an item
# date - date in format dd/mm/yyyy
# date_block_num - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33
# item_name - name of item
# shop_name - name of shop
# item_category_name - name of item category

# This dataset is permitted to be used for any purpose, including commercial use.



"""


"""
In this script, we do preliminary data analysis on our training set.
We also look at the shops.csv, items.csv, and item_categories.csv
We will perform some data visualisation to look at the general trends, and decide on feature engineering strategies

shops.csv contains, for each shop_id, the shop name
items.csv contains, for each item_id, the name and category id
item_categories.csv contains, for each category id, the name of the category

from shops.csv, we 
# identified duplicate shop ids using fuzzy match of shop names
# extracted city and shop type from shop name for each shop id, saved as shop_id_city_type.csv
# plotted bar charts of cities and types that the 60 shops belong to

from items.csv and item_categories.csv, we
# extracted item main category and platform for each item id, saved as items_id_cat_platform.csv
# plotted bar charts of main categories and platforms that the 84 categories and the 21807 items belong to

from training data, we
# keep only the [1e-5, 1-1e-5] quartiles (we use a different interval in the master file) of 'item_cnt_day' and 'item_price' to discard outliers
# identified shops that are open for at least 5 (?) months, and are still open last month
# identified the month each shop first opened

# identified items that are sold for at least 3 months, and are still sold in the last three months
# identified the month each item was first sold, if at all
# satisfying both are 4871 out of 21802 items

# visualised number of items sold 'monthly' in each shop, and in each city and shop type, found strong dependence on month of the year, and some dependence on the year (2013, 2014 or 2015)





"""


# %% Preamble

# Make the output look better
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# default='warn' # ignores warning about dropping columns inplace
pd.options.mode.chained_assignment = None


os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\4_competitive-data-science-predict-future-sales')


# %% load original datasets

# get training set column names
train_cols = pd.read_csv('datasets/sales_train.csv', nrows=0).columns

# import original training dataset
train_df_raw = pd.read_csv(r'datasets\sales_train.csv', low_memory=False)
# train_df_raw.reset_index(inplace=True)
# train_df_raw.drop('index', inplace=True, axis=1)
train_df_raw.shape
# train_df_raw.shape
# (2935849, 6)


# import original test dataset
test_cols = pd.read_csv('datasets/test.csv', nrows=0).columns
test_df = pd.read_csv('datasets/test.csv')
test_df.shape
# (214200, 3)

# %% load shop and item info

shop_id_city_type = pd.read_csv(
    'engineered_datasets/shop_id_city_type.csv', index_col=[0])
items_id_cat_platform = pd.read_csv(
    'engineered_datasets/items_id_cat_platform.csv', index_col=[0])


# %% investigate our data


# %% overall

"""# No nan in the datasets"""

train_df_raw.isnull().sum()
# date              0
# date_block_num    0
# shop_id           0
# item_id           0
# item_price        0
# item_cnt_day      0
# dtype: int64

test_df.isnull().sum()
# ID         0
# shop_id    0
# item_id    0
# dtype: int64

"""# column dtypes: only 'date' is of 'object' type"""

train_df_raw.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 2935849 entries, 0 to 2935848
# Data columns (total 6 columns):
#  #   Column          Dtype
# ---  ------          -----
#  0   date            object
#  1   date_block_num  int64
#  2   shop_id         int64
#  3   item_id         int64
#  4   item_price      float64
#  5   item_cnt_day    float64
# dtypes: float64(2), int64(3), object(1)
# memory usage: 134.4+ MB

"""# number of (shop, item)-ids in training set"""

num_id = 0
for group, frame in train_df_raw.groupby(['shop_id', 'item_id']):
    num_id += 1
print(num_id)
# 414124
"""# there are 60 shops"""
train_df_raw['shop_id'].nunique()
# 60
"""# there are 21807 items"""
train_df_raw['item_id'].nunique()
# 21807


# !!!
"""
encode item_main_category and platform with [log] frequency?
"""

# %% training data


def plots4():
    fig = plt.figure(dpi=150)
    train_df_raw[['item_cnt_day']].hist(
        bins=100, grid=False, ax=plt.gca(), color='skyblue')
    plt.yscale('log')
    ax = plt.gca()
    ax.set_xlabel('Number of products sold')
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(-0.1, 750)
    # plt.savefig('plots/training dataset/item_cnt_day2.png', dpi = 150)

    fig = plt.figure(dpi=150)
    train_df_raw[['item_price']].hist(
        bins=100, grid=False, ax=plt.gca(), color='seagreen')
    plt.yscale('log')
    ax = plt.gca()
    ax.set_xlabel('Item price')
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(-0.1, 75000)
    # plt.savefig('plots/training dataset/item_price2.png', dpi = 150)


# !!!
"""
set range: iten_cnt_day to (0, 750), item_price to (0, 75000)
"""


# !!!
"""# truncation 1"""
train_df = train_df_raw.copy()

train_df = train_df[
    (train_df['item_cnt_day'] > 0)
    & (train_df['item_cnt_day'] < 750)
    & (train_df['item_price'] > 0.01)
    & (train_df['item_price'] < 75000)]


# %% duplicate shops

# !!!
"""
step 2: merge duplicate shops
"""
"""# plot below figures without merging them to see indeed fuzzy_match_pair = [(10, 11), (57, 0), (58, 1)] are mostly likely indeed duplicates"""

fuzzy_match_pair = [(10, 11), (57, 0), (58, 1)]


def shop_duplicates(df):
    df0 = df.copy()
    for pair in fuzzy_match_pair:
        df0['shop_id'].replace(pair[1], pair[0], inplace=True)
    return df0


def shop_duplicates_remove(df):
    df0 = df.copy()
    for pair in fuzzy_match_pair:
        df0 = df0[df0['shop_id'] != pair[1]]
    return df0


shops_dulicate_removed = [shop[1] for shop in fuzzy_match_pair]

train_df = shop_duplicates(train_df)
shop_id_city_type = shop_duplicates_remove(shop_id_city_type)

# %% training data grouped by shop_id

"""# make pivot table for monthly total item count for <each shop>"""


def train_cnt_shop1(df0):
    df = df0.pivot_table(values='item_cnt_day', index='shop_id',
                         columns='date_block_num', aggfunc=[np.nansum])
    df.fillna(0, inplace=True)
    df.reset_index(inplace=True)
    df.columns = df.columns.get_level_values(1)
    df.rename(columns={'': 'shop_id'}, inplace=True)
    df.set_index('shop_id', inplace=True)
    return df


train_cnt_shop = train_cnt_shop1(train_df)

# plots
fig, _ = plt.subplots(nrows=2, ncols=3)
for j in range(6):
    plt.subplot(2, 3, j+1)
    for i in np.arange(10 * j, min(10 * (j + 1), train_cnt_shop.shape[0])):
        train_cnt_shop.iloc[i,].plot()
        ax = plt.gca()
        ax.set_xlabel(None)
        ax.set_xticks([0, 6, 12, 24, 18, 24, 30])
        ax.set_title(None)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.legend()
fig.text(0.5, 0.04, 'Month since 2013-01', ha='center', fontsize=16)
# plt.savefig('plots/training dataset/shops_month/monthly_total_items_sold_each_shop.png', dpi=300)


def scale_row_mean(row):
    row_mean = row.mean()
    row = row / row_mean
    return row


# scale sales by the mean
train_cnt_shop_scaled = train_cnt_shop.apply(scale_row_mean, axis=1)

plt.figure(dpi=150)
for i in np.arange(0, train_cnt_shop.shape[0]):
    if train_cnt_shop_scaled.index[i] in [8, 9, 20, 23, 32, 33, 36, 40]:
        continue
    train_cnt_shop_scaled.iloc[i,].plot(ax=plt.gca(), legend=False)
ax = plt.gca()
ax.set_xlabel('Month since 2013-01')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(0, 4)
# plt.savefig('plots/training dataset/shops_month/monthly_total_items_sold_each_shop_scaled.png')


"""# identify shops that are still open when training period ends"""
"""
some shops are closed and some opened midway
we define a shop to be 'open' if it was open for the last 3 months, and was open last month
"""
# !!!
""" step 3"""


def open_last_month(row):
    # is open if shop was open for at least 3 months, open last month
    if (row > 0).sum() < 3 or (row.iloc[-1:].sum() < 1.):
        row['open_last_month'] = False
    else:
        row['open_last_month'] = True

    return row


train_cnt_shop_still_open = train_cnt_shop.apply(open_last_month, axis=1)
train_cnt_shop_still_open = train_cnt_shop_still_open['open_last_month'].astype(
    int)


def first_open_month(row):
    # find first month with non-zero sales count
    row['first_open_month_num'] = row[row > 0].index[0]
    return row


train_cnt_shop_first_open = train_cnt_shop.apply(first_open_month, axis=1)
train_cnt_shop_first_open = train_cnt_shop_first_open['first_open_month_num'].astype(
    int)


train_cnt_shop_open = pd.merge(
    train_cnt_shop_still_open, train_cnt_shop_first_open, how='inner', left_index=True, right_index=True)

"""# save to csv"""
# train_cnt_shop_open.to_csv('engineered_datasets/train_cnt_shop_open.csv')


"""# monthly total item count for <each !open! shop>"""

"""step 4 """
# train_df_still_open = pd.merge(train_df, train_cnt_shop_still_open, how = 'left', on = 'shop_id')
# train_df_still_open = train_df_still_open[train_df_still_open['open_last_month'] == 1]

# train_cnt_shop_still_open = train_df_still_open.pivot_table(values = 'item_cnt_day', index = 'shop_id', columns = 'date_block_num', aggfunc = [np.nansum])
# train_cnt_shop_still_open.fillna(0, inplace = True)
# train_cnt_shop_still_open.reset_index(inplace = True)
# train_cnt_shop_still_open.columns = train_cnt_shop_still_open.columns.get_level_values(1)
# train_cnt_shop_still_open.rename(columns = {'': 'shop_id'}, inplace = True)
# train_cnt_shop_still_open.set_index('shop_id', inplace = True)

# plots

# def plots6():
# fig = plt.figure(dpi = 300)
shops_dulicate_removed = [shop[1] for shop in fuzzy_match_pair]
fig, _ = plt.subplots(nrows=3, ncols=3)
for j in range(6):
    plt.subplot(2, 3, j+1)
    for i in np.arange(10 * j, min(10 * (j + 1), train_cnt_shop.shape[0])):
        if i in shops_dulicate_removed:
            plot_color = 'lightgrey'
        elif train_cnt_shop_open.loc[i]['open_last_month'] == 1:
            plot_color = 'darkgreen'
        else:
            plot_color = 'lightgrey'
        train_cnt_shop.iloc[i,].plot(color=plot_color)
        ax = plt.gca()
        ax.set_xlabel(None)
        ax.set_xticks([0, 6, 12, 18, 24, 30])
        ax.set_title(None)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
fig.text(0.5, 0.04, 'Month since 2013-01', ha='center', fontsize=16)
# plt.legend()
# plt.savefig('plots/training dataset/shops_month/monthly_total_items_sold_each_shop_STILL_OPEN.png', dpi=150)

shops_dulicate_removed = [shop[1] for shop in fuzzy_match_pair]
plt.figure(dpi=150)
for i in np.arange(0, train_cnt_shop.shape[0]):
    if i in shops_dulicate_removed:
        plot_color = 'lightgrey'
    elif train_cnt_shop_open.loc[i]['open_last_month'] == 1:
        plot_color = 'darkgreen'
    else:
        plot_color = 'lightgrey'
    if train_cnt_shop_scaled.index[i] in [8, 9, 20, 23, 32, 33, 36, 40]:
        continue
    train_cnt_shop_scaled.iloc[i,].plot(ax=plt.gca(), color=plot_color)
ax = plt.gca()
ax.set_xlabel('Month since 2013-01')
ax.set_xticks([0, 6, 12, 18, 24, 30])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(0, 4)
# plt.savefig('plots/training dataset/shops_month/monthly_total_items_sold_each_shop_STILL_OPEN_scaled.png', dpi=150)


""" # auto-correlation"""


shops_dulicate_removed = [shop[1] for shop in fuzzy_match_pair]
j = 1
fig, _ = plt.subplots()
for i in train_cnt_shop.index:
    if i in shops_dulicate_removed:
        plot_color = 'lightgrey'
    elif train_cnt_shop_open.loc[i]['open_last_month'] == 1:
        plot_color = 'darkgreen'
    else:
        plot_color = 'lightgrey'
    plot_acf(train_cnt_shop.loc[i], ax=plt.subplot(
        8, 8, j), label=str(i), color=plot_color)
    j = j + 1
    ax = plt.gca()
    ax.set_xlabel(None)
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
fig.text(0.5, 0.04, 'Lag /month', ha='center', fontsize=16)
plt.tight_layout()
# plt.suptitle('auto-correlation of shops still open at the end of month 33')
# plt.savefig('plots/training dataset/shops_month/auto-correlation_all_shops.png', dpi=300)

shops_dulicate_removed = [shop[1] for shop in fuzzy_match_pair]
j = 1
fig, _ = plt.subplots()
for i in train_cnt_shop.index:
    if i in shops_dulicate_removed:
        plot_color = 'lightgrey'
    elif train_cnt_shop_open.loc[i]['open_last_month'] == 1:
        plot_color = 'darkgreen'
    else:
        plot_color = 'lightgrey'
    plot_pacf(train_cnt_shop.loc[i], ax=plt.subplot(
        8, 8, j), label=str(i), color=plot_color)
    j = j + 1
    ax = plt.gca()
    ax.set_xlabel(None)
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
fig.text(0.5, 0.04, 'Lag /month', ha='center', fontsize=16)
plt.tight_layout()
# plt.suptitle('partial auto-correlation of shops still open at the end of month 33')
# plt.savefig('plots/training dataset/shops_month/partial auto-correlation_all_shops.png', dpi=300)

# !!!
""" 1 month, 12-14 months"""

""" # monthly total item count <by city and type of shop>"""

train_cnt_shop_city_type = pd.merge(
    shop_id_city_type, train_cnt_shop, how='left', left_index=True, right_index=True)

train_cnt_shop_city = train_cnt_shop_city_type.drop(
    ['shop_type', 'shop_id'], axis=1).groupby(['shop_city']).agg(np.nansum)
train_cnt_shop_type = train_cnt_shop_city_type.drop(
    ['shop_city', 'shop_id'], axis=1).groupby(['shop_type']).agg(np.nansum)

# train_cnt_shop_city_summonths = train_cnt_shop_city.sum(axis = 1)
# train_cnt_shop_type_summonths = train_cnt_shop_type.sum(axis = 1)


# scale sales by the mean
train_cnt_shop_city_scaled = train_cnt_shop_city.apply(scale_row_mean, axis=1)
train_cnt_shop_type_scaled = train_cnt_shop_type.apply(scale_row_mean, axis=1)

# plots

fig = plt.figure()
for j in range(4):
    plt.subplot(2, 3, j+1)
    for i in np.arange(8 * j, min(8 * (j + 1), train_cnt_shop_city.shape[0])):
        if train_cnt_shop_city.iloc[i,].name != 'москва':
            train_cnt_shop_city.iloc[i,].plot()
            ax = plt.gca()
            ax.set_xlabel(None)
            ax.set_xticks([0, 6, 12, 18, 24, 30])
            ax.set_title(None)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.legend()
plt.subplot(2, 3, 5)
train_cnt_shop_city.loc['москва'].plot()
fig.text(0.5, 0.04, 'Month since 2013-01', ha='center', fontsize=16)
plt.legend()
# plt.suptitle('Total number of items sold in each city') # title
# plt.savefig('plots/training dataset/shops_month/Total number of items sold in each city.png', dpi=300)


fig = plt.figure(dpi=150)
for i in np.arange(0, train_cnt_shop_city_scaled.shape[0]):
    if train_cnt_shop_city_scaled.iloc[i,].name != 'выездная':
        train_cnt_shop_city_scaled.iloc[i,].plot()
ax = plt.gca()
ax.set_xlabel('Month since 2013-01')
ax.set_xticks([0, 6, 12, 18, 24, 30])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(fontsize=8, loc='upper left', ncol=2)
# plt.suptitle('Total number of items sold in each city') # title
# plt.savefig('plots/training dataset/shops_month/Total number of items sold in each city_scaled.png', dpi=300)


plt.figure(dpi=150)
for j in range(train_cnt_shop_type.shape[0]):
    # plt.subplot(3,3,j+1);
    train_cnt_shop_type.iloc[j,].plot()
ax = plt.gca()
ax.set_xlabel('Month since 2013-01')
ax.set_xticks([0, 6, 12, 18, 24, 30])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
# plt.suptitle('Total number of items sold in each shop type') # title
# plt.savefig('plots/training dataset/shops_month/Total number of items sold in each shop type.png', dpi=150)

plt.figure(dpi=150)
for j in range(train_cnt_shop_type_scaled.shape[0]):
    # plt.subplot(3,3,j+1);
    train_cnt_shop_type_scaled.iloc[j,].plot()
ax = plt.gca()
ax.set_xlabel('Month since 2013-01')
ax.set_xticks([0, 6, 12, 18, 24, 30])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
# plt.suptitle('Total number of items sold in each shop type') # title
# plt.savefig('plots/training dataset/shops_month/Total number of items sold in each shop type_scaled.png', dpi=150)


""" # monthly total item count <by the month (1 - 12)> for <each shop> """

train_df_copy = train_df.copy()
train_df_copy['month_num'] = train_df_copy['date_block_num'] % 12 + 1
train_df_copy['year'] = train_df_copy['date_block_num'] // 12 + 1

# summed over all years for each month


def train_cnt_shop_month_num1(df0):
    df = df0.pivot_table(values='item_cnt_day', index='shop_id',
                         columns='month_num', aggfunc=[np.nansum])
    df.fillna(0, inplace=True)
    df.reset_index(inplace=True)
    df.columns = df.columns.get_level_values(1)
    df.rename(columns={'': 'shop_id'}, inplace=True)
    df.set_index('shop_id', inplace=True)
    return df


train_cnt_shop_month_num = train_cnt_shop_month_num1(train_df_copy)

# plots
# def plots9():
#     plt.figure()
#     for j in range(6) :
#         plt.subplot(2,3,j+1);
#         for i in np.arange(10 * j, min(10 * (j + 1), train_cnt_shop.shape[0])) :
#             train_cnt_shop_month_num.iloc[i,].plot(xlabel = 'month of the year')
#         plt.legend()
#     plt.suptitle('Total number of items sold for each shop') # title

#     plt.savefig('3_Total number of items sold for each shop in each month of the year.png', dpi=600)

# year-wise


def train_cnt_shop_month_year1(df0):
    df = df0.pivot_table(values='item_cnt_day', index='shop_id', columns=[
                         'month_num', 'year'], aggfunc=[np.nansum])
    df.fillna(0, inplace=True)
    df.reset_index(inplace=True)
    df.columns = zip(df.columns.get_level_values(1),
                     df.columns.get_level_values(2))
    df.rename(columns={('', ''): 'shop_id'}, inplace=True)
    df.set_index('shop_id', inplace=True)
    return df


train_cnt_shop_month_year = train_cnt_shop_month_year1(train_df_copy)

# plots

# def plots10():
#     for i in range(train_cnt_shop_month_num.shape[0]) :
#         train_cnt_shop_month_year.loc[i, [(j, 1) for j in np.arange(1,13)]].plot(xlabel = 'month of the year', title = 'Total number of items sold for each shop in 2013', label = '2013')
#     plt.savefig('3_Total number of items sold for each shop in 2013.png', dpi=600)

#     plt.figure()
#     for i in range(train_cnt_shop_month_num.shape[0]) :
#         train_cnt_shop_month_year.loc[i, [(j, 2) for j in np.arange(1,13)]].plot(xlabel = 'month of the year', title = 'Total number of items sold for each shop in 2014', label = '2014')
#     plt.savefig('3_Total number of items sold for each shop in 2014.png', dpi=600)

#     plt.figure()
#     for i in range(train_cnt_shop_month_num.shape[0]) :
#         train_cnt_shop_month_year.loc[i, [(j, 3) for j in np.arange(1,11)]].plot(xlabel = 'month of the year', title = 'Total number of items sold for each shop in 2014', label = '2015')
#     plt.savefig('3_Total number of items sold for each shop in 2015.png', dpi=600)


""" # monthly total item count <by the month (1 - 12)> for all shops """

# train_cnt_shop_month_num.sum().plot(xlabel = 'month of the year', title = 'Total number of items sold in all shops')
# plt.savefig('3_Total number of items sold in all shops in each month of the year.png', dpi=600)

# def month_year_all_shops():
train_cnt_shop_month_year_all_shops = train_cnt_shop_month_year.sum()

train_cnt_shop_month_2013 = train_cnt_shop_month_year_all_shops.loc[[
    (j, 1) for j in np.arange(1, 13)]]
train_cnt_shop_month_2013.rename(index=dict(
    zip(train_cnt_shop_month_2013.index, np.arange(1, 13))), inplace=True)
train_cnt_shop_month_2014 = train_cnt_shop_month_year_all_shops.loc[[
    (j, 2) for j in np.arange(1, 13)]]
train_cnt_shop_month_2014.rename(index=dict(
    zip(train_cnt_shop_month_2014.index, np.arange(1, 13))), inplace=True)
train_cnt_shop_month_2015 = train_cnt_shop_month_year_all_shops.loc[[
    (j, 3) for j in np.arange(1, 11)]]
train_cnt_shop_month_2015.rename(index=dict(
    zip(train_cnt_shop_month_2015.index, np.arange(1, 11))), inplace=True)

fig = plt.figure(dpi=150)
pd.concat([train_cnt_shop_month_2013, train_cnt_shop_month_2014, train_cnt_shop_month_2015], axis=1).rename(
    columns={0: 2013, 1: 2014, 2: 2015}).plot(kind='bar', xlabel='Month', ax=plt.gca(), rot=0)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('Total number of items sold in all shops in each month in each year')
# plt.savefig('plots/training dataset/shops_month/Total number of items sold in all shops in each month in each year.png', dpi=150)

# pd.concat([train_cnt_shop_month_2013, train_cnt_shop_month_2014, train_cnt_shop_month_2015], axis = 1).mean(axis = 1).plot(kind = 'bar', xlabel = 'month')
# plt.title('Yearly-mean of Total number of items sold in all shops in each month')
# plt.savefig('3_Yearly-mean of Total number of items sold in all shops in each month.png', dpi=600)

# !!!
""" # shops: find those that opened last month, and find which month they first opened"""


# %% training data grouped by item_id

"""# make pivot table for monthly total item count for each item"""


def train_cnt_item1(df0):
    df = train_df.pivot_table(values='item_cnt_day', index='item_id',
                              columns='date_block_num', aggfunc=[np.nansum])
    df.fillna(0, inplace=True)
    df.reset_index(inplace=True)
    df.columns = df.columns.get_level_values(1)
    df.rename(columns={'': 'item_id'}, inplace=True)
    df.set_index('item_id', inplace=True)
    return df


train_cnt_item = train_cnt_item1(train_df)

# def plots12():
#     for i in np.arange(1, train_cnt_item.shape[0], 50) :
#         train_cnt_item.iloc[i,].plot(xlabel = 'month since January 2013', title = 'Total number of items sold for each item')
# plt.savefig('4_Total number of items sold for each item.png', dpi=600)

""" # monthly total item count by main category and platform"""

train_cnt_item_cat_platform = pd.merge(
    items_id_cat_platform, train_cnt_item, how='left', on='item_id')

""" group by category"""
train_cnt_item_cat = train_cnt_item_cat_platform.drop(
    ['item_category_id', 'item_id', 'platform'], axis=1).groupby(['item_main_category']).agg(np.nansum)


# plt.figure(dpi = 150)
# for i in range(train_cnt_item_cat.shape[0]) :
#     train_cnt_item_cat.iloc[i,].plot(ax = plt.gca())
# ax = plt.gca()
# ax.set_xlabel('Month since 2013-01')
# ax.set_xticks([0,6,12,18,24,30])
# ax.set_title(None)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.legend(fontsize = 8)
# # plt.savefig('plots/training dataset/items/Total number of items sold in each main category.png', dpi=150)


# scale sales by the mean
train_cnt_item_cat_scaled = train_cnt_item_cat.apply(scale_row_mean, axis=1)

# discard low count items due to high fluctuations
plt.figure(dpi=150)
for i in range(train_cnt_item_cat.shape[0]):
    if train_cnt_item_cat.index[i] in ['билеты', 'гарнитуры/наушники', 'служебные']:
        print(i)
        continue
    train_cnt_item_cat_scaled.iloc[i,].plot(ax=plt.gca())
ax = plt.gca()
ax.set_xlabel('Month since 2013-01')
ax.set_xticks([0, 6, 12, 18, 24, 30])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(fontsize=8, loc='upper left')
# plt.ylim(0,20)
# plt.savefig('plots/training dataset/items/Total number of items sold in each main category_scaled.png', dpi=150)


""" group by platform"""
train_cnt_item_pltform = train_cnt_item_cat_platform.drop(
    ['item_id', 'item_main_category', 'item_category_id'], axis=1).groupby(['platform']).agg(np.nansum)
# scale sales by the mean
train_cnt_item_pltform_scaled = train_cnt_item_pltform.apply(
    scale_row_mean, axis=1)

# discard low count items due to high fluctuations
plt.figure(dpi=150)
for i in range(train_cnt_item_pltform_scaled.shape[0]):
    if train_cnt_item_pltform_scaled.index[i] in ['android', 'mac', 'ps2']:
        continue
    train_cnt_item_pltform_scaled.iloc[i,].plot(ax=plt.gca())
    ax = plt.gca()
    ax.set_xlabel('Month since 2013-01')
    ax.set_xticks([0, 6, 12, 18, 24, 30])
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(fontsize=8)
    plt.ylim(0, 10)
    # plt.savefig('plots/training dataset/items/Total number of items sold on each platform_scaled.png', dpi=150)


""" # auto-correlation"""


fig, _ = plt.subplots()
j = 1
for i in train_cnt_item_cat.index:
    plot_acf(train_cnt_item_cat.loc[i], ax=plt.subplot(3, 5, j), label=str(i))
    plt.legend()
    j = j + 1
    ax = plt.gca()
    ax.set_xlabel(None)
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
fig.text(0.5, 0.04, 'Lag /month', ha='center', fontsize=16)
# plt.savefig('plots/training dataset/items/auto-correlation of main categories.png', dpi=300)

fig, _ = plt.subplots()
j = 1
for i in train_cnt_item_cat.index:
    plot_pacf(train_cnt_item_cat.loc[i], ax=plt.subplot(3, 5, j), label=str(i))
    plt.legend()
    j = j + 1
    ax = plt.gca()
    ax.set_xlabel(None)
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
fig.text(0.5, 0.04, 'Lag /month', ha='center', fontsize=16)
# plt.savefig('plots/training dataset/items/partial_auto-correlation of main categories.png', dpi=300)

# !!!
""" 1 month lag, and 12. not much really"""

""" # item: sold in at least three months, and sold in last month, which month they were first sold if sold at all"""


def sold_last_three_months(row):
    if (row > 0).sum() < 3 or (row.iloc[-3:].sum() < 1.):
        row['sold_last_three_months'] = False
    else:
        row['sold_last_three_months'] = True

    return row


train_cnt_item_still_sold = train_cnt_item.apply(
    sold_last_three_months, axis=1)
train_cnt_item_still_sold = train_cnt_item_still_sold['sold_last_three_months'].astype(
    int)


def first_sold_month(row):
    if len(row[row > 0]) > 0:
        row['first_sold_month_num'] = row[row > 0].index[0]
    else:
        row['first_sold_month_num'] = -1
    return row


train_cnt_item_first_sold = train_cnt_item.apply(first_sold_month, axis=1)
train_cnt_item_first_sold = train_cnt_item_first_sold['first_sold_month_num'].astype(
    int)

train_cnt_item_sold = pd.merge(
    train_cnt_item_still_sold, train_cnt_item_first_sold, how='inner', left_index=True, right_index=True)

# save to csv
# train_cnt_item_sold.to_csv('engineered_datasets/train_cnt_item_sold.csv')

# %% item prices


"""# monthly mean price for each item"""

# make pivot table
train_df_still_sold = pd.merge(
    train_df, train_cnt_item_sold, how='left', on='item_id')
train_df_still_sold = train_df_still_sold[train_df_still_sold['sold_last_three_months'] == 1]

train_item_price = train_df_still_sold.pivot_table(
    values='item_price', index='item_id', columns='date_block_num', aggfunc=[np.nanmean])
train_item_price.reset_index(inplace=True)
train_item_price.columns = train_item_price.columns.get_level_values(1)
train_item_price.rename(columns={'': 'item_id'}, inplace=True)
train_item_price.set_index('item_id', inplace=True)

# scale prices by the means
train_item_price_scaled = train_item_price.apply(scale_row_mean, axis=1)


plt.figure(dpi=150)
for i in np.arange(0, 300):
    train_item_price_scaled.iloc[i,].plot(
        xlabel='Month since 2013-01', ax=plt.gca())
ax = plt.gca()
# ax.set_xlabel(None)
# ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(0, 3)
# plt.savefig('plots/training dataset/items/Mean monthly price for each item STILL SOLD_scaled.png', dpi=150)

""" mean item price of each category/platform"""
train_price_item_cat_platform = pd.merge(
    items_id_cat_platform, train_item_price, how='left', on='item_id')

""" group by category"""
train_price_item_cat = train_price_item_cat_platform.drop(
    ['item_category_id', 'item_id', 'platform'], axis=1).groupby(['item_main_category']).agg(np.nanmean)
# scale prices by the means
train_price_item_cat_scaled = train_price_item_cat.apply(
    scale_row_mean, axis=1)


plt.figure(dpi=150)
for i in range(train_price_item_cat_scaled.shape[0]):
    train_price_item_cat_scaled.iloc[i,].plot(ax=plt.gca())
ax = plt.gca()
ax.set_xlabel('Month since 2013-01')
# ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(0, 3.3)
plt.legend(loc='upper right', fontsize=8, ncol=2)
# plt.savefig('plots/training dataset/items/Mean monthly price for each item STILL SOLD_by_main_category_scaled.png', dpi=150)


"""group by platform"""
train_price_item_pltform = train_price_item_cat_platform.drop(
    ['item_id', 'item_main_category', 'item_category_id'], axis=1).groupby(['platform']).agg(np.nanmean)
# scale prices by the means
train_price_item_pltform_scaled = train_price_item_pltform.apply(
    scale_row_mean, axis=1)


plt.figure(dpi=150)
for i in range(train_price_item_pltform_scaled.shape[0]):
    # if train_price_item_pltform.iloc[i, ].name != 'NA':
    train_price_item_pltform_scaled.iloc[i,].plot(ax=plt.gca())
ax = plt.gca()
ax.set_xlabel('Month since 2013-01')
# ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(loc='upper right', fontsize=8)
# plt.ylim(0,3.3)
# plt.savefig('plots/training dataset/items/Mean monthly price for each item STILL SOLD_by_platform_scaled.png', dpi=150)


""" # auto-correlation"""

fig, _ = plt.subplots()
j = 1
for i in train_price_item_cat.index:
    plot_acf(train_price_item_cat.loc[i],
             ax=plt.subplot(3, 5, j), label=str(i))
    plt.legend()
    j = j + 1
    ax = plt.gca()
    ax.set_xlabel(None)
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
fig.text(0.5, 0.04, 'Lag /month', ha='center', fontsize=16)
# plt.savefig('plots/training dataset/items/auto-correlation of mean price in main categories.png', dpi=300)

fig, _ = plt.subplots()
j = 1
for i in train_price_item_cat.index:
    plot_pacf(train_price_item_cat.loc[i],
              ax=plt.subplot(3, 5, j), label=str(i))
    plt.legend()
    j = j + 1
    ax = plt.gca()
    ax.set_xlabel(None)
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
fig.text(0.5, 0.04, 'Lag /month', ha='center', fontsize=16)
# plt.savefig('plots/training dataset/items/partial_auto-correlation of mean price in main categories.png', dpi=300)

# !!!
""" prices do not change much"""
""" 1 month lag"""


# %%
