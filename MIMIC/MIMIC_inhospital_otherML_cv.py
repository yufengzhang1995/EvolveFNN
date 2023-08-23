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
    parser.add_argument('--n', type = int, default = 6, help = 'number of of bucket or visit')
    parser.add_argument('--f', type = int,  default = 5, help = 'n_folds for cross validation')
    parser.add_argument('--s',  type = int, default = 1234, help = 'the random seed')
    args = parser.parse_args()

    #============= Experiment Configurations============================
    # Step Up Parameters
    dataset_root = './MIMIC_inhospital'
    n_bucket = args.n
    time_step = args.n
    model_name = args.m
    out_root = 'MIMIC_output'
    unique_id = f'model_{model_name}_{n_bucket}_reduced'
    random_state = args.s
    n_folds = args.f



    exp_save_path = os.path.join(out_root, unique_id)
    if not os.path.isdir(out_root):
        os.mkdir(out_root)
    if not os.path.isdir(exp_save_path):
        os.mkdir(exp_save_path)
    print('######################################')
    print('Experiment ID:', unique_id)
    print('######################################')
    
    
    
     
    train_file = os.path.join(dataset_root,f'InHospitalMortality_train_sample_17903_timestep_{time_step}_reduced_features.p') # _reduced_features
    test_file = os.path.join(dataset_root,f'InHospitalMortality_test_sample_3236_timestep_{time_step}_reduced_features.p')
    
    train_dataset = pickle.load(open(train_file,'rb'))
    test_dataset = pickle.load(open(test_file,'rb'))


    X = np.array(train_dataset['variables']).astype(np.float32)
    X_static = np.array(train_dataset['static_variables']).astype(np.float32)
    y = np.array(train_dataset['response'])

    X_test = np.array(test_dataset['variables']).astype(np.float32) 
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_static_test = np.array(test_dataset['static_variables']).astype(np.float32)
    y_test = np.array(test_dataset['response'])
    X_test_all = np.concatenate([X_test,X_static_test],axis = -1)


    rule_data = None
    split_method = 'sample_wise'
    category_info = np.array(train_dataset['category_info']).astype(np.int32)
    num_classes = train_dataset['num_classes']
    feature_names = train_dataset.get('feature_names')
    static_feature_names = train_dataset.get('static_feature_names')
    static_category_info = np.array(train_dataset['static_category_info']).astype(np.int32)



    num_time_varying_features = len(feature_names)

    random_state = 1234

    print('The shape of dynamic training  data:',X.shape)
    print('The shape of static training  data:',X_static.shape)
    print('The shape of testing data:',X_test_all.shape)

    
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
        X_train, X_static_train, y_train, X_val, X_static_val, y_val = utils.split_dynamic_static(ss, X, X_static, y, index=index)
        

        X_train_variant = np.reshape(X_train, (X_train.shape[0], -1))
        X_train_all = np.concatenate([X_train_variant,X_static_train],axis = -1)
        
        # for model validation
        X_val_variant = np.reshape(X_val, (X_val.shape[0], -1))
        X_val_all = np.concatenate([X_val_variant,X_static_val],axis = -1)
        print(f'********* fold {index} ************')
        print('The shape of model training time-varying data:',X_train_all.shape)
        print('The shape of model validation time-varying data:',X_val_all.shape)

        
        if model_name == 'LR':
            classifier = LogisticRegression(random_state=0,penalty='none')
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
        elif model_name == 'GFN':
            classifier = GeneralizedFuzzyClassifier(
             n_classes=2,
             max_steps=800, #100000
             category_info=all_category_info,
             batch_size = 64,
             learning_rate=0.1,
             report_freq = 50, # 50
             patience_step = 500, # 500
             random_state=random_state,
             epsilon_training=False,
             binary_pos_only=True,
             weighted_loss=[1.0, 1.0], #
             split_method='sample_wise',
             verbose=0,
             val_ratio=0.3,
             min_epsilon=0.9,
             init_rule_index = None,
             rule_data = None,
             sparse_regu=1e-5,
            corr_regu=1e-5,
             ) 

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