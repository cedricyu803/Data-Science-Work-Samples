# -*- coding: utf-8 -*-
"""
Created on Sun Aug 01 21:00:00 2021

@author: Cedric Yu
"""

# import matplotlib.pyplot as plt
# import math
# import re

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
 

from tensorflow import random
random.set_seed(0)

# print(random.uniform([1]))
# red_wine = pd.read_csv(r'datasets\winequality-red.csv')

import os
os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\6_nyc-taxi-trip-duration')

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
        num_dense_hidden = hp.Int('num_dense_hidden', min_value = 1, max_value = 10, step = 1)
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
        
        units0 = hp.Int('units0', min_value = 16, max_value = 1024, step = 16)
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
                units = hp.Int('units' + str(l), min_value = 16, max_value = 1024, step = 16)
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

def my_callbacks(early_stopping=False, early_stopping_monitor='val_loss', min_delta=0.0001, patience=100, restore_best_weights=True) : 
    if early_stopping == False : return None
    else :
        early_stopping_scheme = [EarlyStopping(
            monitor = early_stopping_monitor,
            mode = 'min', 
            min_delta = min_delta, # minimium amount of change to count as an improvement
            patience = patience, # how many epochs to wait before stopping
            restore_best_weights = restore_best_weights,
        )]
        # print('early stopping done')
        return early_stopping_scheme

#%% load engineered datasets for model training
# !!!

X_train_encoded4 = pd.read_csv('engineered_datasets/2021-09-06_5/X_train_encoded4.csv', index_col = [0])
X_valid_encoded4 = pd.read_csv('engineered_datasets/2021-09-06_5/X_valid_encoded4.csv', index_col = [0])
X_train_encoded4_scaled = pd.read_csv('engineered_datasets/2021-09-06_5/X_train_encoded4_scaled.csv', index_col = [0])
X_valid_encoded4_scaled = pd.read_csv('engineered_datasets/2021-09-06_5/X_valid_encoded4_scaled.csv', index_col = [0])

y_train_log = pd.read_csv('engineered_datasets/2021-09-06_5/y_train_log.csv', index_col = [0])
y_valid_log = pd.read_csv('engineered_datasets/2021-09-06_5/y_valid_log.csv', index_col = [0])
y_train_log = y_train_log.to_frame()
y_valid_log = y_valid_log.to_frame()

# y_train = pd.read_csv('engineered_datasets/2021-09-06_5/y_train.csv', index_col = [0])
# y_valid = pd.read_csv('engineered_datasets/2021-09-06_5/y_valid.csv', index_col = [0])

###############################################
"""# dataset input and output metadata"""

input_shape = [X_train_encoded4_scaled.shape[1]]

# set default as binary classification
label_cols = y_train_log.shape[1]
# label_cols = y_train.shape[1]
classification_problem = False
# 1 for binary classification or regression, C (int, > 1) for multi-class C classification <<after one-hot encoding>>

# activation function of output layer: sigmoid for binary classification, softmax for multi-class classification, linear for regression
if label_cols == 1 : 
    if classification_problem == True : # binary classification
        output_activation = 'sigmoid'
    else :  # regression
        output_activation = None 
elif (type(label_cols) == int) & (label_cols > 1) : # multi-class classification
        output_activation ='softmax' 

###############################################
"""# hypermodel tuning: finding the best set of hyperparameters"""

###############################################
"""# define the hypermodel """

# specify the loss and metrics to train the model with
my_loss = "mean_squared_error"
my_metrics = 'mean_squared_error'

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
    objective = kt.Objective('val_loss', direction="min"),
    # hyperparameters = hp, # overriding existing hyperparameters
    # `tune_new_entries: False to tune <only> the hyperparameters specified above, True to tune <all other> hyperparameters <not> fixed above
    # tune_new_entries = False,
    max_trials = 30,  # Set to 5 to run quicker, but need 100+ for good results
    # loss="mse", # overriding existing loss
    overwrite = True)

tuner.search_space_summary()

###############################################
"""# specify mini-batch size and epoch, and begin search"""

#!!! ideally we want to do this search with cross validation
# mini-batch size
batch_size= 512
num_epochs = 40

tuner.search(X_train_encoded4_scaled, y_train, 
              validation_data = (X_valid_encoded4_scaled, y_valid),
              batch_size = batch_size,
              epochs = num_epochs
              )

tuner.results_summary()

# !!!
###############################################
"""# pick the best set of hyperparameters"""

best_model = tuner.get_best_models()[0]

best_model.summary()
best_model.get_config()
best_model.optimizer.get_config()
best_model.get_weights()

###############################################
"""# <re-train> the model with the best set of hyperparameters"""

# best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
# # build a hypermodel
# build_my_model = HyperModel(input_shape=input_shape, label_cols=label_cols, output_activation=output_activation, loss=my_loss, metrics=my_metrics)
# # build the model with the best hyperparameters
# my_model = build_my_model.build(best_hyperparameters)

my_model = keras.Sequential([
    layers.Dense(units = 2048, input_shape=input_shape, activation='tanh'), # use tanh
    layers.Dense(units=1, activation=None) 
    ]) 


optimizer = keras.optimizers.Adam(learning_rate = 0.01)
my_model.compile(
    optimizer=optimizer,
    loss="mean_squared_error", # binary cross entropy aka cost function 
    metrics='mean_squared_error' # bindary accuracy as metric
)

my_model.summary()
my_model.get_config()
my_model.optimizer.get_config()

"""# re-train the model with train-validation sets"""

# mini-batch size
batch_size= 512
num_epochs = 1000

# early stopping
early_stopping = True
early_stopping_monitor = "val_loss"
min_delta = 0.00001
patience = 20
restore_best_weights = True


history = my_model.fit(X_train_encoded4_scaled, y_train_log, 
                          validation_data = (X_valid_encoded4_scaled, y_valid_log), 
                          batch_size = batch_size,
                          epochs = num_epochs, 
                          callbacks = my_callbacks(early_stopping, 
                                                    early_stopping_monitor, 
                                                    min_delta, patience, 
                                                    restore_best_weights),
                          verbose = 'auto')


# history = my_model.fit(X_train_encoded4_scaled, y_train, 
#                           validation_data = (X_valid_encoded4_scaled, y_valid), 
#                           batch_size = batch_size,
#                           epochs = num_epochs, 
#                           callbacks = my_callbacks(early_stopping, 
#                                                     early_stopping_monitor, 
#                                                     min_delta, patience, 
#                                                     restore_best_weights),
#                           verbose = 'auto')

history_df = pd.DataFrame(history.history)

history_df.loc[:, ['loss']].plot()
history_df.loc[:, ['val_loss']].plot()
history_df.loc[:, ['binary_accuracy']].plot()
history_df.loc[:, ['val_binary_accuracy']].plot()

plt.figure()
plt.plot(history.history['loss'][15:], color='red', label='loss')
plt.plot(history.history['val_loss'][15:], color='blue', label='val_loss')
plt.legend()
plt.savefig('loss', dpi = 300)

# plt.figure()
# plt.plot(history.history['binary_accuracy'], color='red', label='binary_accuracy')
# plt.plot(history.history['val_binary_accuracy'], color='blue', label='val_binary_accuracy')
# plt.legend()

"""# save the model"""

# default directory is found by os.getcwd()
my_model.save('my_nn_model.h5')

# my_loaded_model = keras.models.load_model('my_model')


"""# finally, evaluate the model performance with the holdout test set """

print(my_model.evaluate(X_train_encoded4_scaled, y_train))
print(my_model.evaluate(X_valid_encoded4_scaled, y_valid))
# print(my_model.evaluate(X_test, y_test))
# [0.1671351194381714, 0.1671351194381714]
# [0.18545366823673248, 0.18545366823673248]


# R2 scores
# metric = tfa.metrics.r_square.RSquare()

# metric.update_state(y_train.squeeze(), my_model.predict(encoded_X_train_scaled).squeeze())
# result = metric.result()
# print(result.numpy())
# metric.update_state(y_valid.squeeze(), my_model.predict(encoded_X_valid_scaled).squeeze())
# result = metric.result()
# print(result.numpy())
# 0.88460743
# 0.8561509

#%% test dataset

test_df_all = pd.read_csv('datasets/test.csv')
X_test_id = test_df_all['id']
del test_df_all

X_test_encoded4 = pd.read_csv('engineered_datasets/2021-09-06_5/X_test_encoded4.csv', index_col = [0])
X_test_encoded4_scaled = pd.read_csv('engineered_datasets/2021-09-06_5/X_test_encoded4_scaled.csv', index_col = [0])


X_test_encoded4_scaled.shape
# (625134, 35)

# predict

pred_log = my_model.predict(X_test_encoded4_scaled)
pred = np.exp(pred_log) - 1.

y_test_predict = pd.Series(pred.squeeze(), index= X_test_id, name='trip_duration')
# all positive

y_test_predict.to_csv('predictions/2021-09-06_5/y_test_predict_nn.csv')

# 0.16321

#%%

# tuner.results_summary()
# Results summary
# Results in .\untitled_project
# Showing 10 best trials
# Objective(name='val_loss', direction='min')

# Trial summary
# Hyperparameters:
# num_dense_hidden: 1
# learning_rate: tanh
# dropout: False
# batch_normalize: False
# kernel_regularizer_which: None
# units0: 1024
# learning_rate_decay: PolynomialDecay
# learning_rate_initial : 0.01
# optimizer_which: Adam
# momentum: 0.0
# W_l1: 0.1
# units1: 592
# units2: 1024
# units3: 16
# units4: 1024
# end_learning_rate: 0.0001
# decay_steps: 950
# decay_power: 0.5
# dropout_rate: 0.1
# W_l2: 0.1
# units5: 1024
# rho: 0.9
# beta1: 0.9
# beta2: 0.999
# units6: 624
# units7: 16
# units8: 672
# units9: 16
# Score: 0.18695037066936493


# Trial summary
# Hyperparameters:
# num_dense_hidden: 1
# learning_rate: tanh
# dropout: False
# batch_normalize: False
# kernel_regularizer_which: None
# units0: 1024
# learning_rate_decay: PolynomialDecay
# learning_rate_initial : 0.01
# optimizer_which: Adam
# momentum: 0.0
# W_l1: 0.1
# units1: 1024
# units2: 384
# units3: 16
# units4: 16
# end_learning_rate: 0.0001
# decay_steps: 950
# decay_power: 0.5
# dropout_rate: 0.1
# W_l2: 0.1
# units5: 1024
# rho: 0.9
# beta1: 0.9
# beta2: 0.999
# units6: 16
# units7: 96
# units8: 288
# units9: 1024
# Score: 0.1870133876800537


# Trial summary
# Hyperparameters:
# num_dense_hidden: 1
# learning_rate: tanh
# dropout: False
# batch_normalize: False
# kernel_regularizer_which: None
# units0: 1024
# learning_rate_decay: PolynomialDecay
# learning_rate_initial : 0.01
# optimizer_which: Adam
# momentum: 0.0
# W_l1: 0.1
# units1: 784
# units2: 1024
# units3: 16
# units4: 16
# end_learning_rate: 0.0001
# decay_steps: 950
# decay_power: 0.5
# dropout_rate: 0.4
# W_l2: 0.001
# units5: 1024
# rho: 0.9
# beta1: 0.9
# beta2: 0.999
# Score: 0.1870221644639969


# Trial summary
# Hyperparameters:
# num_dense_hidden: 1
# learning_rate: tanh
# dropout: False
# batch_normalize: False
# kernel_regularizer_which: None
# units0: 880
# learning_rate_decay: PolynomialDecay
# learning_rate_initial : 0.01
# optimizer_which: Adam
# momentum: 0.0
# W_l1: 0.1
# units1: 752
# units2: 16
# units3: 704
# units4: 608
# end_learning_rate: 0.0001
# decay_steps: 950
# decay_power: 0.5
# dropout_rate: 0.6
# W_l2: 0.001
# units5: 1024
# rho: 0.9
# beta1: 0.9
# beta2: 0.999
# Score: 0.18703097105026245


# Trial summary
# Hyperparameters:
# num_dense_hidden: 1
# learning_rate: tanh
# dropout: False
# batch_normalize: False
# kernel_regularizer_which: None
# units0: 1024
# learning_rate_decay: PolynomialDecay
# learning_rate_initial : 0.01
# optimizer_which: Adam
# momentum: 0.0
# W_l1: 0.1
# units1: 16
# units2: 1024
# units3: 784
# units4: 16
# end_learning_rate: 0.0001
# decay_steps: 950
# decay_power: 1.5
# dropout_rate: 0.1
# W_l2: 0.1
# units5: 1024
# rho: 0.9
# beta1: 0.9
# beta2: 0.999
# units6: 720
# units7: 768
# units8: 256
# units9: 16
# Score: 0.18708838522434235
# Trial summary
# Hyperparameters:
# num_dense_hidden: 1
# learning_rate: tanh
# dropout: False
# batch_normalize: False
# kernel_regularizer_which: None
# units0: 864
# learning_rate_decay: PolynomialDecay
# learning_rate_initial : 0.01
# optimizer_which: Adam
# momentum: 0.0
# W_l1: 0.1
# units1: 496
# units2: 16
# units3: 336
# units4: 16
# end_learning_rate: 0.0001
# decay_steps: 950
# decay_power: 0.5
# dropout_rate: 0.6
# W_l2: 0.1
# units5: 1024
# rho: 0.9
# beta1: 0.9
# beta2: 0.999
# units6: 16
# units7: 704
# units8: 976
# units9: 16
# Score: 0.18708926439285278
# Trial summary
# Hyperparameters:
# num_dense_hidden: 1
# learning_rate: tanh
# dropout: False
# batch_normalize: False
# kernel_regularizer_which: None
# units0: 1024
# learning_rate_decay: PolynomialDecay
# learning_rate_initial : 0.01
# optimizer_which: Adam
# momentum: 0.0
# W_l1: 0.1
# units1: 1024
# units2: 720
# units3: 16
# units4: 1024
# end_learning_rate: 0.0001
# decay_steps: 950
# decay_power: 2.5
# dropout_rate: 0.6
# W_l2: 0.001
# units5: 1024
# rho: 0.9
# beta1: 0.9
# beta2: 0.999
# Score: 0.18709497153759003
# Trial summary
# Hyperparameters:
# num_dense_hidden: 1
# learning_rate: tanh
# dropout: False
# batch_normalize: False
# kernel_regularizer_which: None
# units0: 1024
# learning_rate_decay: PolynomialDecay
# learning_rate_initial : 0.01
# optimizer_which: Adam
# momentum: 0.0
# W_l1: 0.1
# units1: 32
# units2: 16
# units3: 1024
# units4: 1024
# end_learning_rate: 0.0001
# decay_steps: 950
# decay_power: 2.5
# dropout_rate: 0.6
# W_l2: 0.1
# units5: 1024
# rho: 0.9
# beta1: 0.9
# beta2: 0.999
# Score: 0.18709778785705566
# Trial summary
# Hyperparameters:
# num_dense_hidden: 1
# learning_rate: tanh
# dropout: False
# batch_normalize: False
# kernel_regularizer_which: None
# units0: 1024
# learning_rate_decay: PolynomialDecay
# learning_rate_initial : 0.01
# optimizer_which: Adam
# momentum: 0.0
# W_l1: 0.1
# units1: 880
# units2: 320
# units3: 576
# units4: 560
# end_learning_rate: 0.0001
# decay_steps: 950
# decay_power: 2.0
# dropout_rate: 0.1
# W_l2: 0.1
# units5: 1024
# rho: 0.9
# beta1: 0.9
# beta2: 0.999
# Score: 0.18710072338581085
# Trial summary
# Hyperparameters:
# num_dense_hidden: 1
# learning_rate: tanh
# dropout: False
# batch_normalize: False
# kernel_regularizer_which: None
# units0: 1024
# learning_rate_decay: PolynomialDecay
# learning_rate_initial : 0.01
# optimizer_which: Adam
# momentum: 0.0
# W_l1: 0.1
# units1: 1024
# units2: 1024
# units3: 16
# units4: 80
# end_learning_rate: 0.0001
# decay_steps: 950
# decay_power: 2.5
# dropout_rate: 0.4
# W_l2: 0.1
# units5: 1024
# rho: 0.9
# beta1: 0.9
# beta2: 0.999
# units6: 16
# units7: 16
# units8: 1024
# units9: 16
# Score: 0.1871088594198227




# best_model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# layer0 (Dense)               (None, 1024)              36864     
# _________________________________________________________________
# layer1 (Dense)               (None, 1)                 1025      
# =================================================================
# Total params: 37,889
# Trainable params: 37,889
# Non-trainable params: 0
# _________________________________________________________________


# {'name': 'sequential_1',
#  'layers': [{'class_name': 'InputLayer',
#    'config': {'batch_input_shape': (None, 35),
#     'dtype': 'float32',
#     'sparse': False,
#     'ragged': False,
#     'name': 'layer0_input'}},
#   {'class_name': 'Dense',
#    'config': {'name': 'layer0',
#     'trainable': True,
#     'batch_input_shape': (None, 35),
#     'dtype': 'float32',
#     'units': 1024,
#     'activation': 'tanh',
#     'use_bias': True,
#     'kernel_initializer': {'class_name': 'GlorotUniform',
#      'config': {'seed': None}},
#     'bias_initializer': {'class_name': 'Zeros', 'config': {}},
#     'kernel_regularizer': None,
#     'bias_regularizer': None,
#     'activity_regularizer': None,
#     'kernel_constraint': None,
#     'bias_constraint': None}},
#   {'class_name': 'Dense',
#    'config': {'name': 'layer1',
#     'trainable': True,
#     'dtype': 'float32',
#     'units': 1,
#     'activation': 'linear',
#     'use_bias': True,
#     'kernel_initializer': {'class_name': 'GlorotUniform',
#      'config': {'seed': None}},
#     'bias_initializer': {'class_name': 'Zeros', 'config': {}},
#     'kernel_regularizer': None,
#     'bias_regularizer': None,
#     'activity_regularizer': None,
#     'kernel_constraint': None,
#     'bias_constraint': None}}]}


# {'name': 'Adam',
#  'learning_rate': {'class_name': 'PolynomialDecay',
#   'config': {'initial_learning_rate': 0.01,
#    'decay_steps': 950,
#    'end_learning_rate': 0.0001,
#    'power': 0.5,
#    'cycle': False,
#    'name': None}},
#  'decay': 0.0,
#  'beta_1': 0.9,
#  'beta_2': 0.999,
#  'epsilon': 1e-07,
#  'amsgrad': False}










