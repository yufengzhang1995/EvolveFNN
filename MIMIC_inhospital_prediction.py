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


dataset_root = './MIMIC_inhospital'
evolve_type = 'GRU'
time_step = 6
out_root = 'models'
test_type = 'full' 
unique_id = f'Dynamic_FNN_time_step{time_step}_{evolve_type}_test_{test_type}_noCV_reduced'
if test_type == 'full':
    train_file = os.path.join(dataset_root,f'InHospitalMortality_train_sample_17903_timestep_{time_step}_reduced_features.p')
    test_file = os.path.join(dataset_root,f'InHospitalMortality_test_sample_3236_timestep_{time_step}_reduced_features.p')
elif test_type == 'partial':
    N = 1200
    train_file = os.path.join(dataset_root,f'InHospitalMortality_train_sample_{N}_timestep_{time_step}.p')
    test_file = os.path.join(dataset_root,f'InHospitalMortality_test_sample_{N}_timestep_{time_step}.p')
    
exp_save_path = os.path.join(out_root, unique_id)
if not os.path.isdir(out_root):
    os.mkdir(out_root)
if not os.path.isdir(exp_save_path):
    os.mkdir(exp_save_path)
print('######################################')
print('Experiment ID:', unique_id)
print('######################################')    
    
    

train_dataset = pickle.load(open(train_file,'rb'))
test_dataset = pickle.load(open(test_file,'rb'))


X = np.array(train_dataset['variables']).astype(np.float32)
X_static = np.array(train_dataset['static_variables']).astype(np.float32)
y = np.array(train_dataset['response'])

X_test = np.array(test_dataset['variables']).astype(np.float32)
X_static_test = np.array(test_dataset['static_variables']).astype(np.float32)
y_test = np.array(test_dataset['response'])

rule_data = None
split_method = 'sample_wise'
category_info = np.array(train_dataset['category_info']).astype(np.int32)
num_classes = train_dataset['num_classes']
feature_names = train_dataset.get('feature_names')
static_feature_names = train_dataset.get('static_feature_names')
static_category_info = np.array(train_dataset['static_category_info']).astype(np.int32)



num_time_varying_features = len(feature_names)

random_state = 1234

ss_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
X_train, X_static_train, y_train, X_val, X_static_val, y_val = utils.split_dynamic_static(ss_train_test, X, X_static, y, index=0)

print('The shape of dynamic training  data:',X_train.shape)
print('The shape of static training  data:',X_static_train.shape)
print('The shape of validation data:',X_val.shape)
print('The shape of testing data:',X_test.shape)

# device = 'cpu'
max_steps = 10000
n_bucket = X_train.shape[1]
classifier = GeneralizedFuzzyEvolveClassifier(
                evolve_type = evolve_type,
                weighted_loss=[1.0,1.0],
                n_visits = n_bucket,
                report_freq=50,
                patience_step=2000,
                max_steps=max_steps,
                learning_rate=0.1,
                batch_size = 64,
                split_method='sample_wise',
                category_info=category_info,
                static_category_info=static_category_info,
                random_state=random_state,
                verbose=2,
                n_rules=30,
                min_epsilon = 0.9,
                sparse_regu=0,
                corr_regu=0,
    
            )
classifier.fit(X_train, y_train,X_static_train,
          X_val,X_static_val,y_val)
pickle.dump(classifier, open(os.path.join(exp_save_path, f'saved_model.mdl'), 'wb'))
train_metrics = utils.cal_metrics_fbeta(classifier,X_train,X_static_train,y_train)
print('auc aucpr F1score')
print('train')
print(train_metrics)
val_metrics = utils.cal_metrics_fbeta(classifier,X_val,X_static_val,y_val)
print('val')
print(val_metrics)
test_metrics = utils.cal_metrics_fbeta(classifier,X_test,X_static_test,y_test)
print('test')
print(test_metrics)