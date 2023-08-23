import numpy as np
import os
import argparse
import pandas as pd
import pickle
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit



## =============== Models ============================ ##
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
import numpy as np

dtype = torch.float32

class RNNModel(nn.Module):
    def __init__(self, input_dim, static_input_dim, hidden_dim, static_hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.static_fc = nn.Linear(static_input_dim, static_hidden_dim)
        self.fc = nn.Linear(hidden_dim+static_hidden_dim, output_dim)

    def forward(self, x, x_static = None):

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, h0 = self.rnn(x, h0.detach())
        if x_static is not None:
            h1 = self.static_fc(x_static)
            out = out[:, -1, :]
            out = torch.cat([out,h1],axis = -1)
        else:
            out = out[:, -1, :]
        out = self.fc(out)
        return out
    
class LSTMModel(nn.Module):
    def __init__(self, input_dim, static_input_dim, hidden_dim, static_hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()


        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.static_fc = nn.Linear(static_input_dim, static_hidden_dim)
        self.fc = nn.Linear(hidden_dim+static_hidden_dim, output_dim)

    def forward(self, x, x_static = None):

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        if x_static is not None:
            h1 = self.static_fc(x_static)
            out = out[:, -1, :]
            out = torch.cat([out,h1],axis = -1)
        else:
            out = out[:, -1, :]
        out = self.fc(out)
        return out
    
class GRUModel(nn.Module):
    def __init__(self, input_dim, static_input_dim, hidden_dim, static_hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()

        
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)

        self.static_fc = nn.Linear(static_input_dim, static_hidden_dim)
        self.fc = nn.Linear(hidden_dim+static_hidden_dim, output_dim)
    
    def forward(self, x, x_static = None):
        
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, _ = self.gru(x, h0.detach())
        if x_static is not None:
            h1 = self.static_fc(x_static)
            out = out[:, -1, :]
            out = torch.cat([out,h1],axis = -1)
        else:
            out = out[:, -1, :]
        out = self.fc(out)
        return out


## =============== Classifiers ============================ ##
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import copy
import torch.utils.data
import torch.nn.functional as F
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn import preprocessing
from sklearn.utils.multiclass import unique_labels
import utils
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def standardize(features, scaler, category_list):
    if scaler is not None:
        if category_list is not None:
            features_continous = features[:, :, category_list==0]
            scaled_features = features.copy()
            for i in range(features.shape[-2]):
                scaled_features[:, i, category_list==0] = scaler[i].transform(features_continous[:,i,:])
        else:
            scaled_features = scaler.transform(features)
    else:
        scaled_features = features
        
    return scaled_features

def static_standarize(features, scaler, category_list):
    if scaler is not None:
        if category_list is not None:
            features_continous = features[:, category_list==0]
            scaled_features = features.copy()
            scaled_features[:, category_list==0] = scaler.transform(features_continous[:,:])
        else:
            scaled_features = scaler.transform(features)
    else:
        scaled_features = features
        
    return scaled_features

def build_dataset_loader(features, static_features,labels=None, batch_size=1, 
                         scaler=None, static_scaler = None, infinite=False, category_list=None,static_category_list = None):
    """
    Standalize featuers and build data loader for model training and evaluation.

    """
    
    if scaler is not None:
        scaled_features = standardize(features, scaler, category_list)
        if static_scaler is not None:
            scaled_static_features = static_standarize(static_featues, static_scaler, static_category_list)
        else:
            scaled_static_features = static_features
    else:
        scaled_features = features   
        scaled_static_features = static_features

    if labels is not None:
        tensor_features = torch.from_numpy(scaled_features)
        tensor_static_features = torch.from_numpy(scaled_static_features)
        tensor_labels = torch.from_numpy(labels.astype(np.int32))
        dataset = torch.utils.data.TensorDataset(tensor_features.to(device), tensor_static_features.to(device),tensor_labels.to(device))
    else:
        tensor_features = torch.from_numpy(scaled_features)
        tensor_static_features = torch.from_numpy(static_features)
        dataset = torch.utils.data.TensorDataset(tensor_features.to(device),tensor_static_features.to(device))
    
    if infinite:
        data_loader = utils.repeater(torch.utils.data.DataLoader(dataset, int(batch_size), shuffle=True))
    else:
        data_loader = torch.utils.data.DataLoader(dataset, int(batch_size), shuffle=False)
        
    return scaled_features, data_loader







class RNNVariant(BaseEstimator, ClassifierMixin):
    
    def __init__(self, 
                 model_type = 'RNN',
                 weighted_loss=None,
                 report_freq=50,
                 patience_step=500,
                 max_steps=10000,
                 learning_rate=0.01,
                 batch_size=150,
                 split_method='sample_wise',
                 category_info=None,
                 static_category_info = None,
                 random_state=None,
                 verbose=2,
                 val_ratio=0.2,):
        self.model_type = model_type
        self.weighted_loss = weighted_loss
        self.report_freq = report_freq
        self.patience_step = patience_step
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.split_method = split_method
        self.random_state = random_state
        self.verbose = verbose
        self.val_ratio = val_ratio
        self.category_info = category_info
        self.static_category_info = static_category_info
        
    
            
            
    def fit(self, X_train_variant, 
                  X_train_invariant, 
                  y_train, 
                  X_val_variant=None, 
                  X_val_invariant=None,
                  y_val=None):

        torch.manual_seed(self.random_state)
        
        
        
        input_dim = X_train_variant.shape[1]
        self.scaler = []
        for i in range(X_train_variant.shape[-2]):
            self.scaler.append(preprocessing.StandardScaler().fit(X_train_variant[:, i, self.category_info==0]))
            
        # self.static_scaler = preprocessing.StandardScaler().fit(X_train_invariant[:, self.static_category_info==0])
        self.static_scaler = None
        # Build the training data loader. It will generate training batch in an infinite way. 
        _, train_loader = build_dataset_loader(X_train_variant, 
                                              X_train_invariant, 
                                              y_train,self.batch_size, 
                                               scaler=self.scaler, 
                                               static_scaler = self.static_scaler,
                                               infinite=True,
                                                category_list=self.category_info,
                                              static_category_list=self.static_category_info)
        
        # Build the data loader that used to evaluate model's performance on the training data, so it is not infinite.
        scaled_train_X, train_loader_for_eval = build_dataset_loader(X_train_variant, 
                                              X_train_invariant, 
                                              y_train, self.batch_size,
                                            scaler=self.scaler, 
                                            static_scaler = self.static_scaler,
                                            infinite=False,
                                            category_list=self.category_info,
                                            static_category_list=self.static_category_info)
        
        # Build the data loader that used to evaluate model's performance on the validation data, and it is not infinite.
        _, val_loader = build_dataset_loader(X_val_variant, 
                                          X_val_invariant,
                                          y_val,self.batch_size, 
                                          scaler=self.scaler, 
                                             static_scaler = self.static_scaler,
                                             infinite=False,
                                          category_list=self.category_info,
                                          static_category_list=self.static_category_info)

        num_time_varying_features = X_train_variant.shape[-1]
        num_time_invariant_features = X_train_invariant.shape[-1]
        # print('XXXXXX:',X_train_variant.shape)
        # print('XXXXXX:',X_train_invariant.shape)
        # print('num_time_invariant_features:',num_time_invariant_features)
        # Build network
        if self.model_type == 'RNN':
            net = RNNModel(input_dim = num_time_varying_features, 
               static_input_dim = num_time_invariant_features, 
               hidden_dim=32, 
               static_hidden_dim=16, 
               layer_dim=1, 
               output_dim=2)
        elif self.model_type == 'LSTM':
            net = LSTMModel(input_dim = num_time_varying_features, 
               static_input_dim = num_time_invariant_features, 
               hidden_dim=32, 
               static_hidden_dim=16, 
               layer_dim=1, 
               output_dim=2)
        elif self.model_type == 'GRU':
            net = GRUModel(input_dim = num_time_varying_features, 
               static_input_dim = num_time_invariant_features, 
               hidden_dim=32, 
               static_hidden_dim=16, 
               layer_dim=1, 
               output_dim=2)

        net.to(device)
        
        self.initial_weights = copy.deepcopy(net.state_dict())
        # Build loss function and optimizer

        if self.weighted_loss is not None:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.weighted_loss, dtype=dtype, device=device))
        else:
            criterion = nn.CrossEntropyLoss()
            
        
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
        
        
        train_aucpr_list = []
        val_aucpr_list = []
        
        patience = 0
        best_value = 0
        best_step = 0
        best_net = copy.deepcopy(net.state_dict())
        delay_step = -1
        
        for global_step, (inputs,static_inputs, labels) in enumerate(train_loader, 0):
            labels = labels.type(torch.long)
            # Optimize
            optimizer.zero_grad()
            if global_step>100:
                for g in optimizer.param_groups:
                    g['lr'] = self.learning_rate*0.5
                    
            # inputs = torch.permute(inputs, (0, 2, 1))

            out = net(inputs,static_inputs) # Forward pass
            
            # Add regularization term
            cross_entropy =  criterion(out, labels)
            loss = cross_entropy
            loss.backward()        
            
            for param in net.parameters():
                if param.grad is not None:
                    param.grad[torch.isnan(param.grad)] = 0
            
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()

            if (global_step+1) % self.report_freq == 0:
                _, _, _, _, f1_train, auc_train, aucpr_train, _, _ = self._model_testing(net, train_loader_for_eval)
                _, _, _, _, f1_val, auc_val, aucpr_val, _, _ = self._model_testing(net, val_loader)
                train_aucpr_list.append(aucpr_train)
                val_aucpr_list.append(aucpr_val) 
                
        
                if f1_val>best_value:
                    best_value = auc_val
                    best_step = global_step
                    best_net = copy.deepcopy(net.state_dict())
                    patience = 0
    
                else:
                    patience += 1
    
                if patience > self.patience_step//self.report_freq:
                    break
                
                if self.verbose==2:
                    print(f'Step {global_step}, train_auc: {auc_train:.3f}, train_aucpr: {aucpr_train:.3f}, train_f1: {f1_train:.3f}, val_auc: {auc_val:.3f}, val_aucpr: {aucpr_val:.3f}, val_f1: {f1_val:.3f}.')
            if global_step > self.max_steps:
                break

        if self.verbose>=1:
            print(f'The best value is {best_value:.2f} at step {best_step}.')
        
        net.load_state_dict(best_net)
        self.estimator = net
        
        self.classes_ = unique_labels(y_train)
        self.X_ = X_train_variant
        self.y_ = y_train

        return self
    

    def predict(self, X):
        """ Predict the class of the given data samples.
        """
        if self.split_method == 'patient_wise':
            X = X[:, 1:]
            
        # Check is fit had been called
        check_is_fitted(self)

        
        _, test_loader = build_dataset_loader(X, X_static, batch_size=self.batch_size, scaler=self.scaler,
                                              static_scaler = self.static_scaler,
                                           infinite=False, category_list=self.category_info,
                                             static_category_list = self.static_category_info)
            
        pred_list = []
        for i, (inputs, static_inputs) in enumerate(test_loader, 0):
            # inputs = torch.permute(inputs[0], (0, 2, 1))
            x = self.estimator(inputs,static_inputs)
            pred = torch.argmax(x, dim=1)
            pred_list.append(pred)
        pred_list = np.concatenate(pred_list, axis=0)
        return pred_list
    
    
    def predict_proba(self, X,X_static):
        """ Predict the probabilites of belonging to individaul classes of the given data samples.
        """
        if self.split_method == 'patient_wise':
            X = X[:, 1:]
            
        check_is_fitted(self)


        _, test_loader = build_dataset_loader(X,X_static, batch_size=self.batch_size, scaler=self.scaler,
                                           static_scaler = self.static_scaler,
                                           infinite=False, category_list=self.category_info,
                                             static_category_list = self.static_category_info)
            
        prob_list = []
        for i, (inputs, static_inputs) in enumerate(test_loader, 0):
            x = self.estimator(inputs,static_inputs)
            prob = F.softmax(x, dim=1)
            prob_list.append(prob.detach())        
            
        prob_list = np.concatenate(prob_list, axis=0)
        prob_list = np.round(prob_list, 3)
        return prob_list
    
    
    def _model_testing(self, net, test_loader):
        """ Model test.
        
        Parameters
        ----------
        net: A Net object. The network with the best validation performance
        test_loader: finite data_loader for evaluation.
        
        Returns
        -------
        Evaluation metrics includeing accuracy, sensitivity, specificity, precision, f1-score, auc, and aucpr.
        """
        pred_list = []
        label_list = []
        prob_list = []
        for i, (inputs, static_inputs,labels) in enumerate(test_loader, 0):
            # inputs = torch.permute(inputs, (0, 2, 1))
            labels = labels.type(torch.long)
            x = net(inputs,static_inputs)
            
            if torch.sum(torch.isnan(x))>0 or torch.sum(torch.isinf(x))>0:
                import time
                filename = time.strftime("%b_%d_%H_%M_%S", time.localtime())
                params = np.array([self.n_rules, self.batch_size, self.learning_rate, 
                                   self.sparse_regu, self.corr_regu])
                np.save(filename, params)
                                
            prob = F.softmax(x, dim=1)
            pred = torch.argmax(x, dim=1)
            pred_list.append(pred)
            label_list.append(labels)
            prob_list.append(prob.detach())
        
        pred_list = torch.cat(pred_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        prob_list = torch.cat(prob_list, dim=0)
            
        pred = pred_list.numpy()
        labels = label_list.numpy()
        probs = prob_list.numpy()
        probs = np.round(probs, 3)
        
        acc = np.sum(pred == labels)/len(labels)
        sen = np.sum(pred[labels==1])/np.sum(labels)
        spe = np.sum(1-pred[labels==0])/np.sum(1-labels)
        pre = np.sum(pred[labels==1])/(np.sum(pred)+1) # avoid warnings when all samples are classified as negative
        f1 = sklearn.metrics.f1_score(labels, pred)
        
        fpr, tpr, _ = sklearn.metrics.roc_curve(labels+1, probs[:,1], pos_label=2)
        auc = sklearn.metrics.auc(fpr, tpr)
        aucpr = sklearn.metrics.average_precision_score(labels, probs[:,1])
        return acc, sen, spe, pre, f1, auc, aucpr, probs, labels



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
    out_root = 'outputs'
    unique_id = f'Dynamic_{n_bucket}_{visit_type}_hold_off_{hold_off}_{evolve_type}_addmed'
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
    
    
    

    dataset_root = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Code/Yufeng/FNN_evolve/dataset'
    ffile_root = os.path.join(dataset_root,f'{visit_type}_based/UMHS_lab_vital_{n_bucket}_{visit_type}s_h_{hold_off}_o_{observation}_add_cardiac_medication.p')
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
                        'SEX',
                        'med_count']


    num_time_varying_features = len(feature_names)
    num_time_invariant_features = len(static_feature_names)
    static_category_info = np.zeros(num_time_invariant_features) + 2
    static_category_info[-2] = 0
    static_category_info = static_category_info.astype(np.int32)

    time_varying_features = data[:,0:num_time_varying_features*n_bucket]
    time_invariant_features = data[:,-num_time_invariant_features:]


    max_steps = 800
    
    # split into training and testing
    ss_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    X, y, X_test,y_test = utils.split_dataset(ss_train_test, data, labels,index=0)
    X_test = utils.fill_in_missing_value(X_test,X_test)
    X_test_variant,X_test_invariant = utils.handle_data(X_test,num_time_invariant_features,n_bucket)

    # split into three-fold cross validation for hyperparamter tuning
    ss = StratifiedShuffleSplit(n_splits=n_folds, random_state=random_state)

    lr_ls = [0.001,0.003,0.01,0.03,0.1,0.3]
    bs_ls = [16,32,64]
    # lr_ls = [0.01]
    # bs_ls = [64]

    evaluation_name = ['auc','auprc','f1']
    colnames = ['{}_{}'.format(set_name, eval_name) for set_name in ['Train','Val','Test'] for eval_name in evaluation_name]
    row_name_list = []
    row_list = []
    for lr in lr_ls:
        for bs in bs_ls:
            expriment_name = f'lr_{lr}_bs_{bs}'
            row_name_list.append(expriment_name)
            fold_train = np.zeros([n_folds, 3])
            fold_val = np.zeros([n_folds, 3])
            fold_test = np.zeros([n_folds, 3])
            
            show_value_list = []
            for index in range(n_folds): 
                # X_train, y_train, X_val, y_val = utils.split_dataset(ss, X, y, split_method, index=index)
                # X_train = utils.fill_in_missing_value(X_train,X_train)
                # X_val = utils.fill_in_missing_value(X_val,X_val)
                # X_train_variant,X_train_invariant = utils.handle_data(X_train,num_time_invariant_features,n_bucket)
                # X_val_variant,X_val_invariant = utils.handle_data(X_val,num_time_invariant_features,n_bucket)
                # print(f'********* fold {index} ************')
                # print('The shape of training time-varying data:',X_train_variant.shape)
                # print('The shape of training time-invariqant data:',X_train_invariant.shape)
                # print('The shape of validation time-varying data:',X_val_variant.shape)
                
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


                classifier = RNNVariant(
                    model_type = evolve_type,
                    weighted_loss=[1.0,1.0],
                    report_freq=50,
                    patience_step=800,
                    max_steps=max_steps,
                    learning_rate=lr,
                    batch_size = bs,
                    split_method=split_method,
                    category_info=category_info,
                    static_category_info=None,
                    random_state=random_state,
                    verbose=2,
                    val_ratio=0.3,
                )
                # classifier.fit(X_train_variant,X_train_invariant, y_train, X_val_variant,X_val_invariant, y_val,)
                # print('train')
                # train_metrics = utils.cal_metrics(classifier,X_train_variant,X_train_invariant,y_train)
                # fold_train[index,:] = train_metrics
                # print('val')
                # val_metrics = utils.cal_metrics(classifier,X_val_variant,X_val_invariant,y_val)
                # fold_val[index,:] = val_metrics
                # print('test')
                # test_metrics = utils.cal_metrics(classifier,X_test_variant,X_test_invariant,y_test)
                # fold_test[index,:] = test_metrics
                classifier.fit(X_internal_train_variant,X_internal_train_invariant,y_internal_train,X_internal_val_variant,X_internal_val_invariant, y_internal_val,)
                print('train')
                train_metrics = utils.cal_metrics(classifier,X_internal_train_variant,X_internal_train_invariant,y_internal_train)
                fold_train[index,:] = train_metrics
                print('val')
                val_metrics = utils.cal_metrics(classifier,X_internal_val_variant,X_internal_val_invariant,y_internal_val)
                fold_val[index,:] = val_metrics
                print('test')
                test_metrics = utils.cal_metrics(classifier,X_val_variant,X_val_invariant,y_val)
                fold_test[index,:] = test_metrics
            
            show_value_list = utils.show_metrics(fold_train, show_value_list)  
            show_value_list = utils.show_metrics(fold_val, show_value_list)
            show_value_list = utils.show_metrics(fold_test, show_value_list)
            
            eval_series = pd.Series(show_value_list, index=colnames)
            row_list.append(eval_series)


    eval_table = pd.concat(row_list, axis=1).transpose()
    print(eval_table)
    print(row_name_list)
    eval_table.index = row_name_list
    eval_table.to_csv(os.path.join(exp_save_path, 'eval_table_add_med_test.csv'))

if __name__ == '__main__':
    main()