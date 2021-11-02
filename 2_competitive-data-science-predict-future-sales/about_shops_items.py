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

 # Submissions are evaluated by <<root mean squared error (RMSE)>>. 
# <<True target values are clipped into [0,20] range>>

# Submission File

# For each id in the test set, you must predict a <<total number of sales>>. The file should contain a header and have the following format:

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
In this script, we look at the shops.csv, items.csv, and item_categories.csv

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
os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\4_competitive-data-science-predict-future-sales')


#%% about shops

"""# load shop names"""
shop_info_df = pd.read_csv(r'datasets\shops.csv', low_memory=False)


"""duplicated shop ids based on shop names"""
# do a fuzzy match on shop names

from fuzzywuzzy import fuzz

fuzzy_match_shops = np.zeros((len(shop_info_df), len(shop_info_df))) # pairwise fuzzy match matrix
fuzzy_match_pair_shops = []
for jj in range(len(shop_info_df)) : 
    for ii in range(jj) : 
        fuzzy_match_shops[ii, jj] = fuzz.token_sort_ratio(shop_info_df['shop_name'][ii], shop_info_df['shop_name'][jj])
        if fuzzy_match_shops[ii, jj] > 80 : 
            fuzzy_match_pair_shops.append((ii, jj))

"""candidate duplicate shop_id pairs """
print(fuzzy_match_pair_shops)
# [(10, 11), (23, 24), (30, 31), (39, 40), (0, 57), (1, 58)]

"""# not all these pairs are duplicates--- take a closer look"""
for pair in fuzzy_match_pair_shops : 
    print(str(pair))
    print(shop_info_df['shop_name'][pair[0]])
    print(shop_info_df['shop_name'][pair[1]] + '\n')

# (10, 11)
# Жуковский ул. Чкалова 39м?
# Жуковский ул. Чкалова 39м²

# (23, 24)
# Москва ТК "Буденовский" (пав.А2)
# Москва ТК "Буденовский" (пав.К7)
# ^^^ different stores

# (30, 31)
# Москва ТЦ "Перловский"
# Москва ТЦ "Семеновский"
# ^^^ different places

# (39, 40)
# РостовНаДону ТРК "Мегацентр Горизонт"
# РостовНаДону ТРК "Мегацентр Горизонт" Островной
# ^^^ not sure, but I think they are different

# (0, 57)
# !Якутск Орджоникидзе, 56 фран
# Якутск Орджоникидзе, 56

# (1, 58)
# !Якутск ТЦ "Центральный" фран
# Якутск ТЦ "Центральный"

"""
# test_df has entries for 'shop_id': 10, 39, 57, 58
# has no entries for 'shop_id': 11, 40, 0, 1
# we will also look at their monthly sales to decide that, the following are really duplicate pairs
"""

# !!!
fuzzy_match_pair = [(10, 11), (57, 0), (58, 1)]

"""# extract city names and shop types from shop names """

def shop_city_type(row) : 

    row['shop_name'] = row['shop_name'].lower()
    
    # first word is the city name
    row['shop_city'] = re.findall('[\w\-]*\s', row['shop_name'])[0].strip()
    
    # there are several shop types
    shop_types = re.findall('тц|трк|трц|ул.|чс|тк', row['shop_name'])
    if len(shop_types) > 0 : 
        row['shop_type'] = shop_types[0]
    else : 
        row['shop_type'] = 'unknown'
    
    return row

shop_id_city_type = shop_info_df.apply(shop_city_type, axis = 1)
shop_id_city_type = shop_id_city_type[['shop_id', 'shop_city', 'shop_type']]

"""# save to csv"""
shop_id_city_type.to_csv('engineered_datasets/shop_id_city_type.csv')


"""# plots"""

def plots1():
    shop_city = shop_id_city_type[['shop_city']].groupby(['shop_city']).size()
    fig = plt.figure(dpi = 150)
    sns.barplot(x = shop_city.values, y = shop_city.index, color = 'skyblue')
    # plt.yscale('log')
    ax = plt.gca()
    ax.set_ylabel('Shop city')
    ax.set_xlabel(None)
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    # plt.savefig('plots/shops/shop_count_in_city.png', dpi = 150)
    
    shop_type = shop_id_city_type[['shop_type']].groupby(['shop_type']).size()
    fig = plt.figure(dpi = 150)
    sns.barplot(x = shop_type.values, y = shop_type.index, color = 'seagreen')
    # plt.yscale('log')
    ax = plt.gca()
    ax.set_ylabel('Shop type')
    ax.set_xlabel(None)
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    # plt.savefig('plots/shops/shop_count_in_type.png', dpi = 150)

""" translations"""

city_name_rus = ['адыгея',
 'балашиха',
 'волжский',
 'вологда',
 'воронеж',
 'выездная',
 'жуковский',
 'интернет-магазин',
 'казань',
 'калуга',
 'коломна',
 'красноярск',
 'курск',
 'москва',
 'мытищи',
 'новгород',
 'новосибирск',
 'омск',
 'ростовнадону',
 'самара',
 'сергиев',
 'спб',
 'сургут',
 'томск',
 'тюмень',
 'уфа',
 'химки',
 'цифровой',
 'чехов',
 'якутск',
 'ярославль']

city_name_eng = ['adygea',
  'balashikha',
  'Volzhsky',
  'Vologda',
  'voronezh',
  'exit',
  'zhukovsky',
  'online store',
  'kazan',
  'kaluga',
  'Kolomna',
  'Krasnoyarsk',
  'Kursk',
  'Moscow',
  'mytischi',
  'novgorod',
  'novosibirsk',
  'Omsk',
  'Rostovnadon',
  'samara',
  'sergiev',
  'spb',
  'surgut',
  'tomsk',
  'tyumen',
  'ufa',
  'khimki',
  'digital',
  'chekhov',
  'yakutsk',
  'Yaroslavl']

city_name_dict = dict(zip(city_name_rus, city_name_eng))
city_name_dict
# {'адыгея': 'adygea',
#  'балашиха': 'balashikha',
#  'волжский': 'Volzhsky',
#  'вологда': 'Vologda',
#  'воронеж': 'voronezh',
#  'выездная': 'exit',
#  'жуковский': 'zhukovsky',
#  'интернет-магазин': 'online store',
#  'казань': 'kazan',
#  'калуга': 'kaluga',
#  'коломна': 'Kolomna',
#  'красноярск': 'Krasnoyarsk',
#  'курск': 'Kursk',
#  'москва': 'Moscow',
#  'мытищи': 'mytischi',
#  'новгород': 'novgorod',
#  'новосибирск': 'novosibirsk',
#  'омск': 'Omsk',
#  'ростовнадону': 'Rostovnadon',
#  'самара': 'samara',
#  'сергиев': 'sergiev',
#  'спб': 'spb',
#  'сургут': 'surgut',
#  'томск': 'tomsk',
#  'тюмень': 'tyumen',
#  'уфа': 'ufa',
#  'химки': 'khimki',
#  'цифровой': 'digital',
#  'чехов': 'chekhov',
#  'якутск': 'yakutsk',
#  'ярославль': 'Yaroslavl'}

shop_type_rus = ['unknown', 'тк', 'трк', 'трц', 'тц', 'ул.', 'чс']
shop_type_eng = ['unknown', 'mk', 'trk', 'trts', 'mts', 'st.', 'chs']
shop_type_dict = dict(zip(shop_type_rus, shop_type_eng))
shop_type_dict

# {'unknown': 'unknown',
#  'тк': 'mk',
#  'трк': 'trk',
#  'трц': 'trts',
#  'тц': 'mts',
#  'ул.': 'st.',
#  'чс': 'chs'}

#%% about items 

"""# load item names"""

items_df = pd.read_csv(r'datasets\items.csv', low_memory=False)
items_cat_df = pd.read_csv(r'datasets\item_categories.csv', low_memory=False)

# doing a fuzzy match on item names in a for-loop takes forever; don't

"""# extract main categories and platforms from item_category_name """

def categories(row) : 
    
    row['item_category_name'] = row['item_category_name'].lower()
    
    # there are several platforms
    platform = re.findall('pc|ps2|ps3|ps4|psp|psvita|xbox\s360|xbox\sone|android|mac', row['item_category_name'])
    if len(platform) > 0 : 
        row['platform'] = platform[0]
        row['item_main_category'] = re.sub(platform[0], '', row['item_category_name'])
    else : 
        row['platform'] = 'other'
        row['item_main_category'] = row['item_category_name']
    
    
    # row['item_category_name'] = re.sub('\(|\)', '', row['item_category_name'])
    category_split = re.split('\s-\s|\s\(', row['item_main_category'])
    category_split = [x for x in category_split if x ]
    # first word is the city name
    row['item_main_category'] = category_split[0].strip()
    
    return row

items_cat_platform = items_cat_df.apply(categories, axis = 1)
items_cat_platform = items_cat_platform[['item_category_id', 'item_main_category', 'platform']]

""" save to csv"""
items_cat_platform.to_csv('engineered_datasets/items_cat_platform.csv')


"""# plots"""

def plots2():
    items_cat_count = items_cat_platform[['item_main_category']].groupby(['item_main_category']).size()
    fig = plt.figure(dpi = 150)
    sns.barplot(x = items_cat_count.values, y = items_cat_count.index, color = 'skyblue')
    # plt.yscale('log')
    ax = plt.gca()
    ax.set_ylabel('Main category')
    ax.set_xlabel(None)
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    # plt.savefig('plots/items/item_cat_count_in_main_cat.png', dpi = 150)
    
    items_platform_count = items_cat_platform[['platform']].groupby(['platform']).size()
    fig = plt.figure(dpi = 150)
    sns.barplot(x = items_platform_count.values, y = items_platform_count.index, color = 'seagreen')
    # plt.yscale('log')
    ax = plt.gca()
    ax.set_ylabel('Item platform')
    ax.set_xlabel(None)
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    # plt.savefig('plots/items/item_cat_count_in_platform.png', dpi = 150)


""" translations"""

main_cat_rus = ['аксессуары',
 'билеты',
 'гарнитуры/наушники',
 'доставка товара',
 'игровые консоли',
 'игры',
 'карты оплаты',
 'кино',
 'книги',
 'музыка',
 'подарки',
 'программы',
 'служебные',
 'чистые носители',
 'элементы питания']

main_cat_eng = ['accessories',
  'tickets',
  'headsets / headphones',
  'delivery of goods',
  'game consoles',
  'games',
  'payment cards',
  'cinema',
  'books',
  'music',
  'present',
  'programs',
  'service',
  'blank media',
  'batteries']

main_cat_dict = dict(zip(main_cat_rus, main_cat_eng))
main_cat_dict
# {'аксессуары': 'accessories',
#  'билеты': 'tickets',
#  'гарнитуры/наушники': 'headsets / headphones',
#  'доставка товара': 'delivery of goods',
#  'игровые консоли': 'game consoles',
#  'игры': 'games',
#  'карты оплаты': 'payment cards',
#  'кино': 'cinema',
#  'книги': 'books',
#  'музыка': 'music',
#  'подарки': 'present',
#  'программы': 'programs',
#  'служебные': 'service',
#  'чистые носители': 'blank media',
#  'элементы питания': 'batteries'}


"""# correct item names """

# get item name--- not used
def item_name_split(row) : 
    
    row['item_name'] = row['item_name'].lower().strip()
    name_split = re.split('\s\[|\]', row['item_name'])
    name_split = [x for x in name_split if x ]
    row['item_name'] = re.sub('[~!@#$%^&*()\[\]\{\}\\\|\'\";:\/.,<>\?]*', '', name_split[0])
    row['item_name'] = re.sub('[\s]+', ' ', row['item_name']).strip()

    return row

items_df = items_df.apply(item_name_split, axis = 1)
items_df = items_df[['item_id', 'item_category_id']]

items_id_cat_platform = pd.merge(items_df, items_cat_platform, on = 'item_category_id')

"""# save to csv"""
items_id_cat_platform.to_csv(r'items_id_cat_platform.csv')

# plots
def plots3():

    items_main_cat = items_id_cat_platform[['item_main_category']].groupby(['item_main_category']).size()
    fig = plt.figure(dpi = 150)
    sns.barplot(x = items_main_cat.values, y = items_main_cat.index, color = 'skyblue')
    plt.xscale('log')
    ax = plt.gca()
    ax.set_ylabel('Main category')
    ax.set_xlabel(None)
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    # plt.savefig('plots/items/item_id_count_in_main_cat.png', dpi = 150)
    
    items_platform = items_id_cat_platform[['platform']].groupby(['platform']).size()
    fig = plt.figure(dpi = 150)
    sns.barplot(x = items_platform.values, y = items_platform.index, color = 'seagreen')
    plt.xscale('log')
    ax = plt.gca()
    ax.set_ylabel('Platform')
    ax.set_xlabel(None)
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    # plt.savefig('plots/items/item_id_count_in_platform.png', dpi = 150)

# !!!
"""
encode item_main_category and platform with [log] frequency?
"""
