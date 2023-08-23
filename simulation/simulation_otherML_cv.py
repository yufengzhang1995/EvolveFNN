import argparse

import pickle
import os,sys
import numpy as np
import xgboost, os
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import utils
from Classifier import GeneralizedFuzzyEvolveClassifier
from sklearn.svm import SVC  
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from GFN import GeneralizedFuzzyClassifier
from sklearn import tree
from interpret.glassbox import ExplainableBoostingClassifier
from simulation_data_gen import *
def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description = 'Cross validation for LR, SVM, MLP, DT')
    parser.add_argument('--m', default = 'LR', help = 'machine learning model name ')
    parser.add_argument('--n', type = int, default = 10, help = 'number of of bucket or visit')
    parser.add_argument('--f', type = int,  default = 5, help = 'n_folds for cross validation')
    parser.add_argument('--s',  type = int, default = 1234, help = 'the random seed')
    args = parser.parse_args()

    #============= Experiment Configurations============================
    # Step Up Parameters
    n_bucket = args.n
    n_timestamp = args.n
    model_name = args.m
    out_root = 'simulation_outputs'
    unique_id = f'model_{model_name}_{n_bucket}'
    random_state = args.s
    n_folds = args.f
    n_samples = 1000



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
    X_test_variant,X_test_invariant = utils.handle_data(X_test,num_time_invariant_features,n_bucket)
    X_test_variant = np.reshape(X_test_variant, (X_test_variant.shape[0], -1))
    X_test_all = np.concatenate([X_test_variant,X_test_invariant],axis = -1)

    # split into three-fold cross validation for hyperparamter tuning
    ss = StratifiedShuffleSplit(n_splits=n_folds, random_state=random_state)

    evaluation_name = ['auc','auprc','f1',"Sensitivity","Specificity","Accuracy","Precision"]
    colnames = ['{}_{}'.format(set_name, eval_name) for set_name in ['Train','Val'] for eval_name in evaluation_name]

    fold_train = np.zeros([n_folds, 7])
    fold_val = np.zeros([n_folds, 7])
        
    show_value_list = []
    row_list = []
    row_name_list = []
    expriment_name = model_name
    row_name_list.append(expriment_name)
    for index in range(n_folds): 
        X_train, y_train, X_val, y_val = utils.split_dataset(ss, X, y, index=index)
        internal_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
        

        X_train_variant,X_train_invariant = utils.handle_data(X_train,num_time_invariant_features,n_bucket)
        X_train_variant = np.reshape(X_train_variant, (X_train_variant.shape[0], -1))
        X_train_all = np.concatenate([X_train_variant,X_train_invariant],axis = -1)
        
        # for model validation
        X_val_variant,X_val_invariant = utils.handle_data(X_val,num_time_invariant_features,n_bucket)
        X_val_variant = np.reshape(X_val_variant, (X_val_variant.shape[0], -1))
        X_val_all = np.concatenate([X_val_variant,X_val_invariant],axis = -1)
        print(f'********* fold {index} ************')
        print('The shape of model training time-varying data:',X_train_all.shape)
        print('The shape of model validation time-varying data:',X_val_all.shape)

        
        if model_name == 'LR':
            classifier = LogisticRegression(random_state=random_state)
        elif model_name == 'KNN':
            classifier = KNeighborsClassifier(n_neighbors=5)
        elif model_name == 'MLP':
            classifier = MLPClassifier(random_state=random_state, max_iter=800)
        elif model_name == 'NB':
            classifier = GaussianNB()
        elif model_name == 'DT':
            classifier = tree.DecisionTreeClassifier()
        elif model_name == 'EBM':
            classifier = ExplainableBoostingClassifier(random_state=random_state)

        classifier.fit(X_train_all, y_train)
        print('train')
        train_metrics = utils.ML_cal_metrics(classifier,X_train_all, y_train)
        fold_train[index,:] = train_metrics
        print('val')
        val_metrics = utils.ML_cal_metrics(classifier,X_val_all, y_val)
        fold_val[index,:] = val_metrics
        
    show_value_list = utils.show_metrics(fold_train, show_value_list)  
    show_value_list = utils.show_metrics(fold_val, show_value_list)
    
    eval_series = pd.Series(show_value_list, index=colnames)
    row_list.append(eval_series)


    eval_table = pd.concat(row_list, axis=1).transpose()
    print(eval_table)
    print(row_name_list)
    eval_table.index = row_name_list
    eval_table.to_csv(os.path.join(exp_save_path, f'eval_table.csv'))

if __name__ == '__main__':
    main()