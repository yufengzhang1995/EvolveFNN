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


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description = 'Cross validation for MIMIC inhospital mortality prediciton using EvolveFNN')
    parser.add_argument('--evolve_type', default = 'GRU',help = 'RNN or LSTM or GRU')
    parser.add_argument('--n', type = int, default = 6, help = 'number of of bucket or visit')
    parser.add_argument('--f', type = int,  default = 5, help = 'n_folds for cross validation')
    parser.add_argument('--s',  type = int, default = 1234, help = 'the random seed')
    parser.add_argument('--lr',  type = float, default = 0.1, help = 'learning rate')
    parser.add_argument('--bs',  type = int, default = 64, help = 'batch size')
    parser.add_argument('--test_type',  default = 'full', help = 'test: 1200; full: all samples')
    args = parser.parse_args()
    dataset_root = './MIMIC_inhospital'
    #============= Experiment Configurations============================
    time_step = args.n
    evolve_type = args.evolve_type
    test_type = args.test_type
    out_root = 'MIMIC_output'
    
    random_state = args.s
    n_folds = args.f
    lr = args.lr
    bs = args.bs
    unique_id = unique_id = f'Dynamic_Mortiality_FNN_time_step{time_step}_{evolve_type}_test_{test_type}_lr_{lr}_bs_{bs}_reduced'

    report_freq = 50
    patience_step = 2000
    max_steps = 10000
    
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

    ss = StratifiedShuffleSplit(n_splits=n_folds, random_state=random_state)
    
    evaluation_name = ['auc','auprc','f1',"Sensitivity","Specificity","Accuracy","Precision"]
    colnames = ['{}_{}'.format(set_name, eval_name) for set_name in ['Train','Val','Test'] for eval_name in evaluation_name]
    fold_train = np.zeros([n_folds, 7])
    fold_val = np.zeros([n_folds, 7])
    fold_test = np.zeros([n_folds, 7])
    fold_classifiers = []
    
    row_name_list = ['inhospital']
    row_list = []
    expriment_name = 'inhospital'
    show_value_list = []
    for index in range(n_folds):
        X_train, X_static_train, y_train, X_val, X_static_val, y_val = utils.split_dynamic_static(ss, X, X_static, y, index=index)
        

        print('The shape of dynamic training  data:',X_train.shape)
        print('The shape of static training  data:',X_static_train.shape)
        print('The shape of validation data:',X_val.shape)
        print('The shape of testing data:',X_test.shape)


        classifier = GeneralizedFuzzyEvolveClassifier(
                        evolve_type = evolve_type,
                        weighted_loss=[1.0,1.5],
                        n_visits = time_step,
                        report_freq=50,
                        patience_step=2000,
                        max_steps=max_steps,
                        learning_rate=lr ,# 0.1
                        batch_size = bs,# 64
                        split_method='sample_wise',
                        category_info=category_info,
                        static_category_info=static_category_info,
                        random_state=random_state,
                        verbose=2,
                        min_epsilon = 0.9,
                        sparse_regu=1e-3,
                        corr_regu=1e-4,
            
                    )
        classifier.fit(X_train, y_train,X_static_train,
                X_val,X_static_val,y_val)
        fold_classifiers.append(classifier)
        train_metrics = utils.cal_metrics_fbeta(classifier,X_train,X_static_train,y_train)
        fold_train[index,:] = train_metrics
        print('train')
        print(train_metrics)
        val_metrics = utils.cal_metrics_fbeta(classifier,X_val,X_static_val,y_val)
        fold_val[index,:] = val_metrics
        print('val')
        print(val_metrics)
        test_metrics = utils.cal_metrics_fbeta(classifier,X_test,X_static_test,y_test)
        fold_test[index,:] = test_metrics
        print('test')
        print(test_metrics)
        fold_classifiers.append(classifier)
            
    show_value_list = utils.show_metrics(fold_train, show_value_list)  
    show_value_list = utils.show_metrics(fold_val, show_value_list)
    show_value_list = utils.show_metrics(fold_test, show_value_list)
    eval_series = pd.Series(show_value_list, index=colnames)
    row_list.append(eval_series)
    
    eval_table = pd.concat(row_list, axis=1).transpose()
    print(eval_table)
    print(row_name_list)
    eval_table.index = row_name_list
    eval_table.to_csv(os.path.join(exp_save_path, 'eval_table_all.csv'))
    pickle.dump(fold_classifiers, open(os.path.join(exp_save_path, f'saved_model.mdl'), 'wb'))

if __name__ == '__main__':
    main()