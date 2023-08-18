"""
Author: Yufeng Zhang

"""


import numpy as np
import sklearn
import pandas as pd
from itertools import repeat
from sklearn.metrics import confusion_matrix

def convert_label_to_int(x):
    return 0 if x == 'Controls'  else 1

def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

def multiclass_roc_auc_score(truth, pred, average="macro"):
    lb = sklearn.preprocessing.LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    pred = lb.transform(pred)            
    return sklearn.metrics.roc_auc_score(truth, pred, average=average)

def split_dataset(stratified_split, data, labels, index=0):       
    index = list(stratified_split.split(data, labels))[index]
    train_index = index[0]
    test_index = index[1]
    # print('test_indx top 10:',sorted(test_index)[:10])
    X_train, X_test = np.take(data, train_index, axis=0), np.take(data, test_index, axis=0)
    y_train, y_test = np.take(labels, train_index, axis=0), np.take(labels, test_index, axis=0) 

    return X_train.astype(np.float32),y_train.astype(np.int32), \
            X_test.astype(np.float32),y_test.astype(np.int32)
            
def split_dynamic_static(stratified_split, data, static_data,labels, index=0) :
    index = list(stratified_split.split(data, labels))[index]
    train_index = index[0]
    test_index = index[1]
    # print('test_indx top 10:',sorted(test_index)[:10])
    X_train, X_test = np.take(data, train_index, axis=0), np.take(data, test_index, axis=0)
    X_static_train,X_static_test = np.take(static_data, train_index, axis=0), np.take(static_data, test_index, axis=0)
    y_train, y_test = np.take(labels, train_index, axis=0), np.take(labels, test_index, axis=0) 

    return X_train.astype(np.float32),X_static_train.astype(np.float32), y_train.astype(np.int32), \
            X_test.astype(np.float32),X_static_test.astype(np.float32),y_test.astype(np.int32)          
            
            
            
            
def fill_in_missing_value(features_A,features_B):
    m_list = np.nanmedian(features_A, axis=0)
    m_mat = np.stack([m_list]*features_B.shape[0],axis=0)
    features_B[pd.isnull(features_B)] = m_mat[pd.isnull(features_B)]
    return features_B


def create_RNN_grid():
    param_grid = {
                  'learning_rate': loguniform(5e-3, 1e-1),
                  'batch_size':[16,32,64],
                }
    return param_grid

def handle_data(data,num_time_invariant_features,n_bucket):
    data_variant = data[:,:-num_time_invariant_features]
    data_variant = data_variant.reshape(data_variant.shape[0],n_bucket,-1)
    data_invariant = data[:,-num_time_invariant_features:]
    return data_variant,data_invariant


def cal_metrics(classifier,X_test_variant,X_test_invariant,y_test):
    probs= classifier.predict_proba(X_test_variant,X_test_invariant)
    predictions = probs[:,-1] >= 0.5
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test, probs[:,1])
    f1_scores = 2*recall*precision/(recall+precision)
    recall = sklearn.metrics.recall_score(y_test, predictions)
    precision = sklearn.metrics.precision_score(y_test, predictions, zero_division=0)
    specificity = sklearn.metrics.recall_score(1-y_test, 1-predictions)
    accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
    # f1 = sklearn.metrics.f1_score(y_test, predictions)
    # print('Best threshold: ', thresholds[np.argmax(f1_scores)])
    # print('Best F1-Score: ', np.max(f1_scores))
    aucpr = sklearn.metrics.average_precision_score(y_test, probs[:,1])
    auc = sklearn.metrics.roc_auc_score(y_test, probs[:,1])
    print(f'acc:{accuracy:4f}; pre :{precision:.4f}; recall:{recall:.4f}; spec:{specificity:.4f}; auc:{auc:.4f}; auprc: {aucpr:.4f}; f1 : {max(f1_scores):.4f}')
    return np.array([auc,aucpr,max(f1_scores)])


def cal_metrics_fbeta(classifier,X_test_variant,X_test_invariant,y_test,beta = 1):
    probs = classifier.predict_proba(X_test_variant,X_test_invariant)
    predictions = probs[:,-1] >= 0.4
    TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()
    Sensitivity = TP/(TP + FN)
    Specificity = TN/(TN + FP) 
    Accuracy = (TN + TP)/(TN+TP+FN+FP)
    Precision = TP/(TP + FP)
    fbeta = (1+beta**2)*((Precision*Sensitivity )/(beta**2 * Precision+Sensitivity))
    aucpr = sklearn.metrics.average_precision_score(y_test, probs[:,1])
    auc = sklearn.metrics.roc_auc_score(y_test, probs[:,1])
    print(f'Auc:{auc:.4f}; Auprc: {aucpr:.4f}; F1 : {fbeta:.4f}; Sensitivity :{Sensitivity }; Specificity: {Specificity}; Accuracy : {Accuracy}; Precision :{Precision}')
    return np.array([auc,aucpr,fbeta,Sensitivity,Specificity,Accuracy,Precision])
    











def ML_cal_metrics(classifier,X_test,y_test,beta = 1):
    probs= classifier.predict_proba(X_test)
    predictions = probs[:,-1] >= 0.5
    TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()
    Sensitivity = TP/(TP + FN)
    Specificity = TN/(TN + FP) 
    Accuracy = (TN + TP)/(TN+TP+FN+FP)
    Precision = TP/(TP + FP)
    fbeta = (1+beta**2)*((Precision*Sensitivity )/(beta**2 * Precision+Sensitivity))
    aucpr = sklearn.metrics.average_precision_score(y_test, probs[:,1])
    auc = sklearn.metrics.roc_auc_score(y_test, probs[:,1])
    print(f'Auc:{auc:.4f}; Auprc: {aucpr:.4f}; F1 : {fbeta:.4f}; Sensitivity :{Sensitivity }; Specificity: {Specificity}; Accuracy : {Accuracy}; Precision :{Precision}')
    return np.array([auc,aucpr,fbeta,Sensitivity,Specificity,Accuracy,Precision])


def show_metrics(metrics, show_value_list=None):
    """ Calculate the average and standard deviation from multiple repetitions and format them.
    """
    eval_m, eval_s = np.nanmean(metrics, 0), np.nanstd(metrics,0)
    for i in range(eval_m.shape[0]):
        show_value_list.append('{:.3f} ({:.3f})'.format(eval_m[i], eval_s[i]))
    return show_value_list

def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input,dim = -1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
        
        
        
def GFN_split_dataset(stratified_split, data, labels, split_method, index=0):        
    """ Split the dataset into the training set and test set.

    Parameters:
    ---------- 
    stratified_split : An instance of the object sklearn.model_selection.StratifiedShuffleSplit.
    data : np.ndarray. A np.array of features with a shape (the number of samples, the number of features).
    labels : np.dnarray. A np.ndarray of labels with a shape (the number of samples,).
    split_method : Str. It indicates how the train/val/test data split should be performed.
        Options include 'patient_wise', and 'sample_wise'. 'sample_wise' is the regular split. 
        For 'patient_wise', data samples from the same patient should be put into the same data set.
    index : Int. The index of the repetition.

    Returns:
    ---------- 
    X_train : np.ndarray with a shape (the number of training samples, the number of features). Training features.
    y_train : np.ndarray with a shape (the number of training samples,). Training labels.
    X_test : np.ndarray with a shape (the number of training samples, the number of features). Testing features.
    y_test : np.ndarray with a shape (the number of training samples,). Testing features.
    
    """
    if split_method == 'patient_wise':
        uids = data[:, 0]        
        uids_HT_VAD_set = set(uids[labels==1])
        uids_too_well_set = set(uids[labels==0]).difference(uids_HT_VAD_set)
        uids_HT_VAD_arr = np.array(list(uids_HT_VAD_set))
        uids_too_well_arr = np.array(list(uids_too_well_set))
        
        uids = np.concatenate([uids_HT_VAD_arr, uids_too_well_arr], axis=0)
        uids_label = np.concatenate([np.ones(uids_HT_VAD_arr.shape[0]), 
                                     np.zeros(uids_too_well_arr.shape[0])], axis=0)
    
        index = list(stratified_split.split(uids, uids_label))[index]
        
        uids_train = np.take(uids, index[0], axis=0)
        uids_test = np.take(uids, index[1], axis=0)
    
        train_index = [i for i in range(data.shape[0]) if data[i,0] in uids_train]
        test_index = [i for i in range(data.shape[0]) if data[i,0] in uids_test]
        
    elif split_method == 'sample_wise':
        # Split the data for nested cross-validation on the training set
        index = list(stratified_split.split(data, labels))[index]
        train_index = index[0]
        test_index = index[1]
    
    else:
        raise NotImplementedError
        
    X_train, X_test = np.take(data, train_index, axis=0), np.take(data, test_index, axis=0)
    y_train, y_test = np.take(labels, train_index, axis=0), np.take(labels, test_index, axis=0)  
        
    return X_train.astype(np.float32), y_train.astype(np.int32), X_test.astype(np.float32), y_test.astype(np.int32)


def GFN_standardize(features, scaler, category_list):
    if scaler is not None:
        if category_list is not None:
            features_continous = features[:, category_list==0]
            scaled_features = features.copy()
            scaled_features[:, category_list==0] = scaler.transform(features_continous)
        else:
            scaled_features = scaler.transform(features)
    else:
        scaled_features = features
        
    return scaled_features
