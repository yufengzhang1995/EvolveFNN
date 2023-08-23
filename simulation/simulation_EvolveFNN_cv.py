from __future__ import absolute_import, division, print_function

import numpy as np
import os
import argparse
import pickle
import os,sys
import numpy as np
import pandas as pd
import pickle
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


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description = 'Hyper-parameter tuning for dynamic Fuzzy Neural Network model of simulated data')
    parser.add_argument('--evolve_type', default = 'GRU',help = 'RNN or LSTM or GRU')
    parser.add_argument('--n', type = int, default = 10, help = 'number of of bucket or visit')
    parser.add_argument('--f', type = int,  default = 5, help = 'n_folds for cross validation')
    parser.add_argument('--s',  type = int, default = 1234, help = 'the random seed')
    args = parser.parse_args()

    #============= Experiment Configurations============================
    n_timestamp = args.n
    evolve_type = args.evolve_type
    out_root = 'simulation_outputs'
    unique_id = f'Dynamic_FNN_{n_timestamp}_{evolve_type}'
    random_state = args.s
    n_folds = args.f

    n_samples = 1000

    report_freq = 50
    patience_step = 600
    max_steps = 800


    exp_save_path = os.path.join(out_root, unique_id)
    if not os.path.isdir(out_root):
        os.mkdir(out_root)
    if not os.path.isdir(exp_save_path):
        os.mkdir(exp_save_path)
    print('######################################')
    print('Experiment ID:', unique_id)
    print('######################################')
    

    time_series_data,static_data,labels = generate_simluated_time_series_data(n_samples,n_timestamp,mislabel = None,random_state=42)

    split_method = 'sample_wise'
    category_info = np.zeros([time_series_data.shape[-1]]).astype(np.int32)
    num_classes = 2
    feature_names = ['x0','x1','x2','x3','x4','x5','x6']
    static_feature_names = ['static_x1','static_x2']


    num_time_varying_features = len(feature_names)
    num_time_invariant_features = len(static_feature_names)
    static_category_info = np.zeros(num_time_invariant_features) 
    static_category_info[0] = 2
    static_category_info = static_category_info.astype(np.int32)

    time_varying_features = time_series_data
    time_invariant_features = static_data
    data = np.concatenate([time_series_data.reshape((n_samples,-1)),static_data],axis =1 )


    # split into training and testing
    ss_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    X, y, X_test,y_test = utils.split_dataset(ss_train_test, data, labels, index=0)
    X_test = utils.fill_in_missing_value(X_test,X_test)
    X_test_variant,X_test_invariant = utils.handle_data(X_test,num_time_invariant_features,n_timestamp)


    ss = StratifiedShuffleSplit(n_splits=n_folds, random_state=random_state)

    lr_ls = [0.003,0.01,0.03,0.1,0.3]
    bs_ls = [16,32,64]
    evaluation_name = ['auc','auprc','f1',"Sensitivity","Specificity","Accuracy","Precision"]
    colnames = ['{}_{}'.format(set_name, eval_name) for set_name in ['Train','Val','Test'] for eval_name in evaluation_name]
    row_name_list = []
    row_list = []
    all_classifiers = {}
    for lr in lr_ls:
        for bs in bs_ls:
            expriment_name = f'lr_{lr}_bs_{bs}'
            row_name_list.append(expriment_name)
            fold_train = np.zeros([n_folds, 7])
            fold_val = np.zeros([n_folds, 7])
            fold_test = np.zeros([n_folds, 7])
            fold_classifiers = []
            
            show_value_list = []
            for index in range(n_folds): 
                X_train, y_train, X_val, y_val = utils.split_dataset(ss, X, y, index=index)
                internal_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
                
                # for model training and early stop
                X_internal_train, y_internal_train, X_internal_val, y_internal_val = utils.split_dataset(internal_train_test, X_train, y_train, index=0)
                X_internal_train = utils.fill_in_missing_value(X_internal_train,X_internal_train)
                X_internal_val = utils.fill_in_missing_value(X_internal_val,X_internal_val)
                X_internal_train_variant,X_internal_train_invariant = utils.handle_data(X_internal_train,num_time_invariant_features,n_timestamp)
                X_internal_val_variant,X_internal_val_invariant = utils.handle_data(X_internal_val,num_time_invariant_features,n_timestamp)
                
                # for model validation
                X_val = utils.fill_in_missing_value(X_val,X_val)
                X_val_variant,X_val_invariant = utils.handle_data(X_val,num_time_invariant_features,n_timestamp)
                print(f'********* fold {index} ************')
                print('The shape of model training time-varying data:',X_internal_train_variant.shape)
                print('The shape of model early stopping time-varying data:',X_internal_val_variant.shape)
                print('The shape of model validation time-varying data:',X_val_variant.shape)


                classifier = GeneralizedFuzzyEvolveClassifier(
                                evolve_type = evolve_type,
                                n_visits = n_timestamp,
                                n_rules = 30,
                                weighted_loss=[1.0,1.0],
                                report_freq=report_freq,
                                patience_step=patience_step,
                                max_steps=max_steps,
                                learning_rate=lr,
                                batch_size = bs,
                                split_method=split_method,
                                category_info=category_info,
                                static_category_info=static_category_info,
                                random_state=random_state,
                                verbose=2,
                                min_epsilon = 0.9,
                                sparse_regu=1e-3,
                                corr_regu=1e-4,
                    
                            )
                classifier.fit(X_internal_train_variant, y_internal_train,X_internal_train_invariant,X_internal_val_variant,X_internal_val_invariant, y_internal_val,)
                print('train')
                train_metrics = utils.cal_metrics_fbeta(classifier,X_internal_train_variant,X_internal_train_invariant,y_internal_train,2)
                fold_train[index,:] = train_metrics
                print('val')
                val_metrics = utils.cal_metrics_fbeta(classifier,X_internal_val_variant,X_internal_val_invariant,y_internal_val,2)
                fold_val[index,:] = val_metrics
                print('test')
                test_metrics = utils.cal_metrics_fbeta(classifier,X_val_variant,X_val_invariant,y_val,2)
                fold_test[index,:] = test_metrics
                fold_classifiers.append(classifier)
            
            show_value_list = utils.show_metrics(fold_train, show_value_list)  
            show_value_list = utils.show_metrics(fold_val, show_value_list)
            show_value_list = utils.show_metrics(fold_test, show_value_list)
            
            eval_series = pd.Series(show_value_list, index=colnames)
            row_list.append(eval_series)
            all_classifiers[expriment_name] = fold_classifiers

    eval_table = pd.concat(row_list, axis=1).transpose()
    print(eval_table)
    print(row_name_list)
    eval_table.index = row_name_list
    eval_table.to_csv(os.path.join(exp_save_path, 'eval_table_all.csv'))
    pickle.dump(all_classifiers, open(os.path.join(exp_save_path, f'saved_model.mdl'), 'wb'))






    

if __name__ == '__main__':
    main()