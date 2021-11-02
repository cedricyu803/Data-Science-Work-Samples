# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 21:00:00 2021

@author: Cedric Yu
"""

# import matplotlib.pyplot as plt
# import math
# import re

#!!! save model mid fitting and afterwards. Wrap it for sklearn pipeline

"""
We use the standard convention, where intances in a dataset runs in row, and features run in columns

# implement all we know about neural network on various example problems (binary/multi-class classification, regression)

# use keras_tuner to build a model with tunable hyperparameters

https://keras.io/api/keras_tuner/
https://keras.io/guides/keras_tuner/getting_started/
https://keras.io/api/keras_tuner/hyperparameters/


save and load a model



# most important hyperparameters to tune
# most important: learning rate, 
# then momentum beta (good default is 0.9), mini-batch size, number of hidden units,
# then number of layers and learning rate decay
# (beta1, beta2, epsilon) no need to tune really

# DO NOT do a grid search
# DO try random values
# because you may not know in advance which hyperparameters are more important
# use coarse-to-fine scheme
# re-test hyperparameters once in a while

# choosing the right scale to sample hyperparameters
# number of units and number of layers: reasonable to sample uniformly
# learning rate: uniformly in log scale
# 1-beta=0.1, 0.01, ...

# bias and variance (Google for definitions)
# high bias = high training set error
# high variance = high dev set error (compared to training set error); sensitive to change of training data point
# This assumes optimal (Bayes) error of 0%. Counter-example: blurry images for cat photo classsification

# basic recipe for machine learning
# 1. high bias (training error)?
# 2. <<bigger network>>, train longer, search for another neural network architecture
# 3. repeat until bias is low
# 4. high variance (dev error) (can we generalise)?
# 5. <<more data>>, regularisation, search for another neural network architecture
# 6. repeat until variance is low
# 7. done =)

Summary: 
    # 

"""


#%% Preamble

import pandas as pd
# Make the output look better
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.options.mode.chained_assignment = None  # default='warn' # ignores warning about dropping columns inplace
# pd.set_option('display.width', 1000)

import numpy as np
import h5py
import seaborn as sn
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

from matplotlib.colors import ListedColormap
cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])


#%% tensorflow.keras

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa # import tfa which contains metrics for regression
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from keras_tuner import HyperModel
from keras_tuner import Objective
 

from tensorflow import random
random.set_seed(0)

# print(random.uniform([1]))
# red_wine = pd.read_csv(r'datasets\winequality-red.csv')


#%% HyperModel
"""# define the hypermodel with HyperModel class in keras_tuner"""

class HyperModel(HyperModel) : 
    def __init__(self, input_shape, label_cols, output_activation, loss, metrics):
        self.input_shape = input_shape
        self.label_cols = label_cols
        self.output_activation = output_activation
        self.loss = loss
        self.metrics = metrics
    
    def build(self, hp):
        
        #####################################################    
        """# hyperparameters for layer definition"""
        
        # number of <hidden> layers
        num_dense_hidden = hp.Int('num_dense_hidden', min_value = 1, max_value = 3, step = 1)
        # activation function for hidden layers
        hidden_activation = hp.Choice('learning_rate', values = ['relu', 'tanh'], default = 'relu')
        # dropout
        dropout = hp.Boolean("dropout", default = False)
        if dropout == True : 
            dropout_rate = hp.Float('dropout_rate', min_value = 0.1, max_value = 0.6, step = 0.1, default = 0.2)
        # batch normalisation
        batch_normalize = hp.Boolean("batch_normalize", default = False)
        # regulariser for kernel W. we do not consider bias and activation regularisers
        kernel_regularizer_which = hp.Choice('kernel_regularizer_which', values = ['None', 'l2', 'l1', 'l1_l2'], default = 'None')
        if kernel_regularizer_which == 'None' : 
            kernel_regularizer = None
        elif kernel_regularizer_which == 'l2' : 
            W_l2 = hp.Choice('W_l2', values = [0.1, 1e-2, 1e-3], default = 1e-2)
            kernel_regularizer = regularizers.l2(l2 = W_l2)
        elif kernel_regularizer_which == 'l1' : 
            W_l1 = hp.Choice('W_l1', values = [0.1, 1e-2, 1e-3], default = 1e-2)
            kernel_regularizer = regularizers.l1(l1 = W_l1)
        elif kernel_regularizer_which == 'l1_l2' : 
            W_l1 = hp.Choice('W_l1', values = [0.1, 1e-2, 1e-3], default = 1e-2)
            W_l2 = hp.Choice('W_l2', values = [0.1, 1e-2, 1e-3], default = 1e-2)
            kernel_regularizer = regularizers.l1_l2(l1 = W_l1, l2 = W_l2)
        
        #####################################################    
        """# layer definition"""
        
        model = keras.Sequential()
        
        units0 = hp.Int('units0', min_value = 16, max_value = 64, step = 16)
        model.add(keras.layers.Dense(units = units0, 
                                      input_shape = self.input_shape, 
                                      activation = hidden_activation, 
                                      kernel_regularizer = kernel_regularizer,
                                      name = 'layer0'))
        if dropout == True : 
            model.add(layers.Dropout(rate = dropout_rate, 
                                      name = 'dropout0'))
        if batch_normalize == True : 
            model.add(layers.BatchNormalization(name = 'batch_normalize0'))
            
        
        if num_dense_hidden > 1 : 
            for l in np.arange(1, num_dense_hidden) : 
                units = hp.Int('units' + str(l), min_value = 16, max_value = 64, step = 16)
                model.add(keras.layers.Dense(units = units, 
                                              activation = hidden_activation, 
                                              kernel_regularizer = kernel_regularizer,
                                              name = 'layer'+str(l)))
                
                if dropout == True : 
                    model.add(layers.Dropout(rate = dropout_rate, 
                                      name = 'dropout' + str(l)))
                if batch_normalize == True : 
                    model.add(layers.BatchNormalization(name = 'batch_normalize' + str(l)))
        
        
        model.add(layers.Dense(units = self.label_cols, 
                                activation = self.output_activation,
                                kernel_regularizer = kernel_regularizer,
                                name = 'layer' + str(num_dense_hidden)))
        
        #####################################################
        """# learning rate decay schedule """
        
        # we only consider constant, exponential decay and power law decay
        learning_rate_decay = hp.Choice('learning_rate_decay', values = ['None', 'ExponentialDecay', 'PolynomialDecay'], default = 'None')
        learning_rate_initial = hp.Choice('learning_rate_initial ', values = [1e-2, 1e-3, 1e-4], default = 0.01)
        if learning_rate_decay == 'None' : 
            learning_rate_schedule = learning_rate_initial
        elif learning_rate_decay == 'ExponentialDecay' : 
            learning_rate_decay_rate = hp.Fixed('learning_rate_decay_rate', value = 0.96)
            decay_steps = hp.Int('decay_steps', min_value = 50, max_value = 1000, step = 100)
            learning_rate_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = learning_rate_initial, 
                decay_steps = decay_steps, 
                decay_rate = learning_rate_decay_rate)
        elif learning_rate_decay == 'PolynomialDecay' : 
            end_learning_rate = hp.Fixed('end_learning_rate', value = 0.0001)
            decay_steps = hp.Int('decay_steps', min_value = 50, max_value = 1000, step = 100)
            decay_power = hp.Float('decay_power', min_value = 0.5, max_value = 2.5, step = 0.5)
            learning_rate_schedule = keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate = learning_rate_initial,
                decay_steps = decay_steps,
                end_learning_rate = end_learning_rate,
                power = decay_power)
    
        #####################################################
        """# model optimiser"""
        
        # we only consider SGD, RMSprop and Adam with default betas
        optimizer_which = hp.Choice('optimizer_which', values = ['SGD', 'RMSprop', 'Adam'])
        if optimizer_which == 'SGD' : 
            momentum = hp.Choice('momentum', values = [0., 0.9])
            optimizer = keras.optimizers.SGD(learning_rate = learning_rate_schedule)
        elif optimizer_which == 'RMSprop' : 
            rho = hp.Fixed('rho', value = 0.9)
            momentum = hp.Choice('momentum', values = [0., 0.9])
            optimizer = keras.optimizers.RMSprop(learning_rate = learning_rate_schedule, rho = rho, momentum = momentum)
        elif optimizer_which == 'Adam' : 
            beta1 = hp.Fixed('beta1', value = 0.9)
            beta2 = hp.Fixed('beta2', value = 0.999)
            optimizer = keras.optimizers.Adam(learning_rate = learning_rate_schedule, beta_1 = beta1, beta_2 = beta2)
        
        
        #####################################################
        """# model optimiser definition and compilation """
        
        model.compile(optimizer = optimizer,
                      loss = self.loss,
                      metrics = self.metrics)
        
        # model.summary()
        
        return model

#     elif my_learning_rate_decay == 'PiecewiseConstantDecay' : 
#         my_learning_rate_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
#         boundaries = my_piecewise_decay_boundaries, 
#         values = my_piecewise_decay_values)

#     elif my_learning_rate_decay == 'InverseTimeDecay' : 
#         my_learning_rate_schedule = keras.optimizers.schedules.InverseTimeDecay(
#         initial_learning_rate = my_learning_rate_initial, 
#         decay_steps = my_decay_steps, 
#         decay_rate = my_learning_rate_decay_rate)

#%% early stopping

def my_callbacks(early_stopping=False, early_stopping_monitor="val_auc_1", min_delta=0.001, patience=100, restore_best_weights=True) : 
    if early_stopping == False : return None
    else :
        early_stopping_scheme = [EarlyStopping(
            monitor = early_stopping_monitor,
            mode = 'max',
            min_delta = min_delta, # minimium amount of change to count as an improvement
            patience = patience, # how many epochs to wait before stopping
            restore_best_weights = restore_best_weights,
        )]
        # print('early stopping done')
        return early_stopping_scheme

#%% pre-processing training and validation datasets from train.csv

###############################################
"""# load dataset """

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

###############################################
"""# separate train_df into features X and target y"""
"""# train-validation-test split"""
"""# DO NOT touch the test set to tune the model or during cross validation!!! """

from sklearn.model_selection import train_test_split
X_train_valid = train_df.drop('compliance', axis=1)
y_train_valid = train_df['compliance']
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, random_state = 0, test_size = 0.1)

###############################################
"""# pre-processing with ColumnTransformer and custom estimators"""

# import pre-processing custom estimators and ColumnTransformer
import assignment_4_preprocessing as pp

# pick useful features and define target of the train-validation set
X_train = pp.drop_cols(X_train)
X_valid = pp.drop_cols(X_valid)

# apply pre-processing
OH_X_train = pp.train_preprocess(X_train)
OH_X_valid = pp.not_train_preprocess(X_valid)

# checked that all dtypes are int64 or float64
# OH_X_train.apply(dtype,axis=0).unique()
# OH_X_valid.apply(dtype,axis=0).unique()

# scaler
OH_X_train_scaled = pp.train_scaler(OH_X_train)
OH_X_valid_scaled = pp.non_train_scaler(OH_X_valid)


y_train = y_train.to_numpy()
y_train = y_train.reshape(y_train.shape[0], 1)
y_valid = y_valid.to_numpy()
y_valid = y_valid.reshape(y_valid.shape[0], 1)

###############################################
"""# dataset input and output metadata"""

input_shape = [OH_X_train_scaled.shape[1]]

# set default as binary classification
label_cols = y_train.shape[1]
classification_problem = True
# 1 for binary classification or regression, C (int, > 1) for multi-class C classification <<after one-hot encoding>>

# activation function of output layer: sigmoid for binary classification, softmax for multi-class classification, linear for regression
if label_cols == 1 : 
    if classification_problem == True : # binary classification
        output_activation = 'sigmoid'
    else :  # regression
        output_activation = None 
elif (type(label_cols) == int) & (label_cols > 1) : # multi-class classification
        output_activation ='softmax' 

#%% 
###############################################
"""# hypermodel tuning: finding the best set of hyperparameters"""

###############################################
"""# define the hypermodel """

# specify the loss and metrics to train the model with
my_loss = "binary_crossentropy"
my_metrics = keras.metrics.AUC()

hypermodel = HyperModel(input_shape=input_shape, label_cols=label_cols, output_activation=output_activation, loss=my_loss, metrics=my_metrics)

###############################################
"""# if desired, choose a subset of hyperparameters to search or fix"""

# hp = kt.HyperParameters()

# to override e.g. the `learning_rate` parameter with our own selection of choices
# hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

# to fix a hyperparameter
# hp.Fixed("learning_rate", value=1e-4)

###############################################
"""# call the tuner"""

tuner = kt.tuners.BayesianOptimization(
    hypermodel,
    objective = Objective('val_auc_1', direction = "max"), 
    # hyperparameters = hp, # overriding existing hyperparameters
    # `tune_new_entries: False to tune <only> the hyperparameters specified above, True to tune <all other> hyperparameters <not> fixed above
    # tune_new_entries = False,
    max_trials = 100,  # Set to 5 to run quicker, but need 100+ for good results
    # loss="mse", # overriding existing loss
    overwrite = True)

tuner.search_space_summary()

###############################################
"""# specify mini-batch size and epoch, and begin search"""

#!!! ideally we want to do this search with cross validation
# mini-batch size
batch_size= 256
num_epochs = 50

tuner.search(OH_X_train_scaled, y_train, 
              validation_data = (OH_X_valid_scaled, y_valid),
              batch_size = batch_size,
              epochs = num_epochs
              )

tuner.results_summary()

###############################################
"""# pick the best set of hyperparameters"""

best_model = tuner.get_best_models()[0]

best_model.summary()
best_model.get_config()
best_model.optimizer.get_config()
best_model.get_weights()

###############################################
"""# <re-train> the model with the best set of hyperparameters"""

best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
# build a hypermodel
build_my_model = HyperModel(input_shape=input_shape, label_cols=label_cols, output_activation=output_activation, loss=my_loss, metrics=my_metrics)
# build the model with the best hyperparameters
my_model = build_my_model.build(best_hyperparameters)


"""# re-train the model with train-validation sets"""

# mini-batch size
batch_size= 256
num_epochs = 100

# early stopping
early_stopping = True
early_stopping_monitor = "val_auc_1"
min_delta = 0.001
patience = 50
restore_best_weights = True

history = my_model.fit(OH_X_train_scaled, y_train, 
                          validation_data = (OH_X_valid_scaled, y_valid), 
                          batch_size = batch_size,
                          epochs = num_epochs, 
                          callbacks = my_callbacks(early_stopping, 
                                                    early_stopping_monitor, 
                                                    min_delta, patience, 
                                                    restore_best_weights),
                          verbose = 'auto')

history_df = pd.DataFrame(history.history)

history_df.loc[:, ['loss']].plot()
history_df.loc[:, ['val_loss']].plot()
history_df.loc[:, ['binary_accuracy']].plot()
history_df.loc[:, ['val_binary_accuracy']].plot()

plt.figure()
plt.plot(history.history['loss'], color='red', label='loss')
plt.plot(history.history['val_loss'], color='blue', label='val_loss')
plt.legend()

plt.figure()
plt.plot(history.history['binary_accuracy'], color='red', label='binary_accuracy')
plt.plot(history.history['val_binary_accuracy'], color='blue', label='val_binary_accuracy')
plt.legend()

"""# save the model"""

# default directory is found by os.getcwd()
my_model.save('my_model')

# my_loaded_model = keras.models.load_model('my_model')


"""# finally, evaluate the model performance with the holdout test set """

print(my_model.evaluate(OH_X_train_scaled, y_train))
print(my_model.evaluate(OH_X_valid_scaled, y_valid))
# print(my_model.evaluate(X_test, y_test))
# [0.19499675929546356, 0.8083586096763611]
# [0.197722390294075, 0.8155434727668762]






#%% model scores

from sklearn.metrics import roc_curve, auc

y_valid_predict_proba = my_model.predict(OH_X_valid_scaled)

fpr, tpr, thresholds  = roc_curve(y_valid, y_valid_predict_proba)
auc = auc(fpr, tpr)

print('RCO-AUC scores: \n')
print('Deep neural network: {:.3f}\n'.format(auc))
# RCO-AUC scores: 
# Deep neural network: 0.815

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr, tpr, lw=3, label='Deep neural network ROC curve (area = {:0.2f})'.format(auc))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()

#%% feature importances

# clf_gb_feature_importances = pd.Series(clf_gb.feature_importances_, index=OH_X_train.columns).sort_values(ascending=False)

# print(clf_gb_feature_importances.head(10))

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
OH_X_pred_scaled = pp.non_train_scaler(OH_X_pred)

my_model.predict(OH_X_pred_scaled)
# array([[0.14503077],
#        [0.0319974 ],
#        [0.07514236],
#        ...,
#        [0.09425337],
#        [0.09425337],
#        [0.68026644]], dtype=float32)

y_test_predict_proba = pd.Series(my_model.predict(OH_X_pred_scaled).squeeze(), index= test_df_all['ticket_id'], name='compliance')








