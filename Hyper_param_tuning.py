"""
Author: Yufeng Zhang

"""

import argparse

import pickle
import os,sys
import numpy as np
import xgboost, os
import pandas as pd
import pickle
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
import utils
from Classifier import GeneralizedFuzzyEvolveClassifier

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description = 'Hyper-parameter tuning for dynamic Fuzzy Neural Network model')
    parser.add_argument('--evolve_type', default = 'GRU',help = 'RNN or LSTM or GRU')
    parser.add_argument('--visit_type',default = 'bucket', help = 'Bucket or visit based?')
    parser.add_argument('--n', type = int, default = 4, help = 'number of of bucket or visit')
    parser.add_argument('--h', type = int, default = 180, help = 'hold_off period')
    parser.add_argument('--o', type = int, default = 360, help = 'observation period')
    parser.add_argument('--f', type = int,  default = 5, help = 'n_folds for cross validation')
    parser.add_argument('--s',  type = int, default = 1234, help = 'the random seed')
    args = parser.parse_args()

    #============= Experiment Configurations============================
    n_bucket = args.n
    evolve_type = args.evolve_type
    hold_off = args.h
    observation = args.o
    visit_type = args.visit_type
    out_root = 'new_outputs'
    unique_id = f'Dynamic_FNN_{n_bucket}_{visit_type}_hold_off_{hold_off}_observation_{observation}_{evolve_type}_add_dynamic_med'
    random_state = args.s
    n_folds = args.f

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
    
    
    

    dataset_root = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Code/Yufeng/FNN_evolve/dataset'
    # ffile_root = os.path.join(dataset_root,f'{visit_type}_based/UMHS_lab_vital_{n_bucket}_{visit_type}s_h_{hold_off}_o_{observation}.p')
    ffile_root = os.path.join(dataset_root,f'{visit_type}_based/UMHS_lab_vital_{n_bucket}_{visit_type}s_h_{hold_off}_o_{observation}_add_dynamic_cardiac_medication.p')
    dataset = pickle.load(open(ffile_root,'rb'))

    dataset = pickle.load(open(ffile_root,'rb'))
    data = np.array(dataset['variables'])
    labels = np.array(dataset['response'])

    rule_data = None
    split_method = 'sample_wise'
    category_info = dataset['category_info']
    num_classes = dataset['num_classes']
    feature_names = dataset.get('feature_names')
    static_feature_names = ['diabetes',
                        'hypertension',
                        'renalfailure',
                        'obesity',
                        'copd',
                        'anemia',
                        'sleepdisorder',
                        'Heart_History',
                        'Smoker',
                        'Alcohol',
                        'Drug',
                        'CURR_AGE_OR_AGE_DEATH',
                        'SEX']


    num_time_varying_features = len(feature_names)
    num_time_invariant_features = len(static_feature_names)
    static_category_info = np.zeros(num_time_invariant_features) + 2
    static_category_info[-2] = 0
    # static_category_info[-3] = 0
    # static_category_info[-1] = 0
    static_category_info = static_category_info.astype(np.int32)

    time_varying_features = data[:,0:num_time_varying_features*n_bucket]
    time_invariant_features = data[:,-num_time_invariant_features:]

    
    # split into training and testing
    ss_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    X, y, X_test,y_test = utils.split_dataset(ss_train_test, data, labels, index=0)
    X_test = utils.fill_in_missing_value(X_test,X_test)
    X_test_variant,X_test_invariant = utils.handle_data(X_test,num_time_invariant_features,n_bucket)

    # split into three-fold cross validation for hyperparamter tuning
    ss = StratifiedShuffleSplit(n_splits=n_folds, random_state=random_state)

    lr_ls = [0.01,0.03,0.1]
    bs_ls = [16,32,64]
    
    
    # lr_ls = [0.1]
    # bs_ls = [64]


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
                X_internal_train_variant,X_internal_train_invariant = utils.handle_data(X_internal_train,num_time_invariant_features,n_bucket)
                X_internal_val_variant,X_internal_val_invariant = utils.handle_data(X_internal_val,num_time_invariant_features,n_bucket)
                
                # for model validation
                X_val = utils.fill_in_missing_value(X_val,X_val)
                X_val_variant,X_val_invariant = utils.handle_data(X_val,num_time_invariant_features,n_bucket)
                print(f'********* fold {index} ************')
                print('The shape of model training time-varying data:',X_internal_train_variant.shape)
                print('The shape of model early stopping time-varying data:',X_internal_val_variant.shape)
                print('The shape of model validation time-varying data:',X_val_variant.shape)


                classifier = GeneralizedFuzzyEvolveClassifier(
                                evolve_type = evolve_type,
                                n_visits = n_bucket,
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
                                sparse_regu=1e-5,
                                corr_regu=1e-5,
                    
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
