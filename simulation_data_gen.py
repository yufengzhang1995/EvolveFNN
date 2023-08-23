from __future__ import absolute_import, division, print_function

import numpy as np
import os
import sklearn
from sklearn import model_selection 
import argparse
import pickle
import os,sys
import numpy as np
import xgboost, os
import pandas as pd
import pickle
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
import utils
from Classifier import GeneralizedFuzzyEvolveClassifier
import matplotlib.pyplot as plt

def generate_simluated_time_series_data(n_samples,n_timestamp,mislabel = None,random_state=1234):
  np.random.seed(random_state)
  x0 = 2**0.5*np.random.randn(n_samples, 1)
  x1 = 3**0.5*np.random.randn(n_samples, 1) + 5
  x2 = 5**0.5*np.random.randn(n_samples, 1) - 1
  x3 = 2**0.5*np.random.randn(n_samples, 1) + 1
  x4 = np.random.randn(n_samples, 1) - 2
  x_noise = np.random.randn(n_samples, 2)
  dynamic_features = np.concatenate([x0, x1, x2, x3, x4, x_noise], axis=-1)

  x1_static = np.expand_dims(np.random.randint(2, size=n_samples), axis=-1)
  x2_static = np.expand_dims(np.random.randint(10, size=n_samples), axis=-1)
  static_features = np.concatenate([x1_static,x2_static], axis=-1)

  timestamps = np.arange(n_timestamp)
  time_series_data = np.zeros((n_samples,n_timestamp,dynamic_features.shape[-1]))
  for i in range(n_samples):
      input_i = dynamic_features[i,:]
      input_i = input_i[:, np.newaxis] + np.cos((2 * input_i[:, np.newaxis] + timestamps[np.newaxis, :]))
      time_series_data[i] = input_i.T

  x0_last = time_series_data[:,-1,0].reshape((-1,1))
  x1_last = time_series_data[:,-1,1].reshape((-1,1))
  x2_last = time_series_data[:,-1,2].reshape((-1,1))
  x3_last = time_series_data[:,-1,3].reshape((-1,1))
  x4_last = time_series_data[:,-1,4].reshape((-1,1))

  rules = [np.logical_and.reduce([x1_last<3.8, x2_last>-2,x1_static == 1], axis=0),
              np.logical_and.reduce([x1_last>6.3, x2_last>-2,x2_static > 6], axis=0),
              np.logical_and.reduce([x0_last<1, x3_last>2], axis=0),
              np.logical_and.reduce([x2_last>0, x4_last>-1,x1_static == 1], axis=0),
              np.logical_and.reduce([x0_last<1, x4_last>-1.5,x2_static > 6], axis=0)]
    
  for i in range(len(rules)):
      print('Rule {}: {:.2f}%'.format(i, np.sum(rules[i])/n_samples*100))

  labels = np.logical_or.reduce(rules, axis=0)[:,0]
  if mislabel is not None:
      one_array = labels[labels==1]
      mutated_one_array = np.where(np.random.random(one_array.shape) < mislabel, False, one_array)
      labels[labels==1] = mutated_one_array


  categorical_info = np.zeros([dynamic_features.shape[1]]) 
  print('Positive samples: {:.2f}%'.format(np.sum(labels)/dynamic_features.shape[0]*100))
  return time_series_data,static_features, labels

def split_data_into_K_fold(n_samples,n_split):
  fold_taskname = np.empty(shape=(n_split, 3), dtype=object)

  idx_all = sorted(range(n_samples))
  for i_split, idx in enumerate(model_selection.KFold(5, shuffle=False).split(idx_all)):
      fold_taskname[i_split][2] = idx[-1]
  for i_split in range(n_split):
      fold_taskname[i_split][1] = fold_taskname[(i_split + 1) % n_split][2]
      fold_taskname[i_split][0] = np.asarray(sorted(set(idx_all).difference(fold_taskname[i_split][1]).difference(fold_taskname[i_split][2])))

  print(fold_taskname[0][0].shape, fold_taskname[0][1].shape, fold_taskname[0][2].shape)
  return fold_taskname

def generagte_train_val_test_from_fold(time_series_data, labels, fold_taskname):
  train_X = np.take(time_series_data,fold_taskname[0][0],axis = 0)
  train_X = train_X.reshape((int(n_samples*0.6),-1))
  train_y = np.take(labels,fold_taskname[0][0])

  test_X = np.take(time_series_data,fold_taskname[0][2],axis = 0)
  test_X = test_X.reshape((int(n_samples*0.2),-1))
  test_y = np.take(labels,fold_taskname[0][2])

  val_X = np.take(time_series_data,fold_taskname[0][1],axis = 0)
  val_X = test_X.reshape((int(n_samples*0.2),-1))
  val_y = np.take(labels,fold_taskname[0][1])

  return train_X, train_y, test_X, test_y,val_X,val_y