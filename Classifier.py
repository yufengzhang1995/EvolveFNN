"""
Author: Yufeng Zhang

"""

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
from Network import Evolve_module


dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_transform(x):
    return np.log(x + 1)

def Winsorizing(data):
    Q1 = np.quantile(data,0.25)
    Q3 = np.quantile(data,0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 0.5 *IQR
    upper_bound = Q3 + 0.5 *IQR

    winsorized_data = np.where(data < lower_bound, lower_bound, data)
    winsorized_data = np.where(data > upper_bound, upper_bound, winsorized_data)
    return winsorized_data  

def standardize(features, scaler, category_list):
    if scaler is not None:
        if category_list is not None:
            features_continous = features[:,:,category_list==0]
            scaled_features = features.copy()
            for i in range(features.shape[1]):
                scaled_features[:,i,category_list==0] = scaler[i].transform(features_continous[:,i,])
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

def build_dataset_loader(features, labels=None, batch_size=1, 
                         scaler=None, infinite=False, category_list=None,
                        X_static = None,static_category_info = None, static_scaler = None):

    
    if scaler is not None:
        scaled_features = standardize(features, scaler, category_list)
    else:
        scaled_features = features  
    
    if X_static is not None:
        if static_scaler is not None:
            scaled_static_features = static_standarize(X_static, static_scaler, static_category_info)
        else:
            scaled_static_features = X_static
        

    if labels is not None:
        if X_static is not None:
            tensor_features = torch.from_numpy(scaled_features).permute(0,2,1)
            tensor_static_features = torch.from_numpy(scaled_static_features)
            tensor_labels = torch.from_numpy(labels.astype(np.int32))
            dataset = torch.utils.data.TensorDataset(tensor_features.to(device), tensor_labels.to(device), tensor_static_features.to(device))
        else:
            tensor_features = torch.from_numpy(scaled_features).permute(0,2,1)
            tensor_labels = torch.from_numpy(labels.astype(np.int32))
            dataset = torch.utils.data.TensorDataset(tensor_features.to(device), tensor_labels.to(device))
            
    else:
        if X_static is not None:
            tensor_static_features = torch.from_numpy(scaled_static_features)
            tensor_features = torch.from_numpy(scaled_features).permute(0,2,1)
            dataset = torch.utils.data.TensorDataset(tensor_features.to(device), tensor_static_features.to(device))
        else:
            tensor_features = torch.from_numpy(scaled_features).permute(0,2,1)
            dataset = torch.utils.data.TensorDataset(tensor_features.to(device))
    
    if infinite:
        data_loader = utils.repeater(torch.utils.data.DataLoader(dataset, int(batch_size), shuffle=True))
    else:
        data_loader = torch.utils.data.DataLoader(dataset, int(batch_size), shuffle=False)
        
    
    
    
    if X_static is not None:
        return scaled_features,scaled_static_features, data_loader
    else:
        return scaled_features,None,data_loader



class GeneralizedFuzzyEvolveClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 weighted_loss=None,
                 epsilon_training=False,
                 init_rule_index=None,
                 binary_pos_only=False,
                 n_rules=20,
                 n_visits = 4,
                 evolve_type = 'GRU',
                 n_classes=2,
                 report_freq=50,
                 patience_step=500,
                 max_steps=10000,
                 learning_rate=0.01,
                 batch_size=150,
                 min_epsilon=0.1,
                 sparse_regu=0,
                 corr_regu=0,
                 category_info=None,
                 static_category_info = None,
                 split_method='sample_wise',
                 winsorized = True,
                 random_state=None,
                 verbose=0,
                 val_ratio=0.2,
                 rule_data=None):
        
        self.weighted_loss = weighted_loss
        self.epsilon_training = epsilon_training
        self.init_rule_index = init_rule_index
        self.binary_pos_only = binary_pos_only
        self.n_rules = n_rules
        self.n_classes = n_classes
        self.report_freq = report_freq
        self.patience_step = patience_step
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.min_epsilon = min_epsilon
        self.sparse_regu = sparse_regu
        self.corr_regu = corr_regu
        self.category_info = category_info
        self.static_category_info = static_category_info
        self.split_method = split_method
        self.random_state = random_state
        self.verbose = verbose
        self.val_ratio = val_ratio
        self.rule_data = rule_data
        self.n_visits = n_visits
        self.evolve_type = evolve_type
        self.winsorized = winsorized
        
        if self.n_classes>2 and self.binary_pos_only:
            raise ValueError('binary_pos_only should be set to true only in a binary classification task.')
        
        if self.init_rule_index is not None and self.rule_data is None:
            raise ValueError('init_rule_index is given but rule_data is None.')
            
        if self.n_classes>2 and self.rule_data is not None:
            raise ValueError('the current design and implementation of rule_data only support binary classification.')
            
            
    def fit(self, X_train, 
                  y_train, 
                  X_train_static = None,
                  X_val = None, 
                  X_val_static = None,
                  y_val = None):
        torch.manual_seed(self.random_state)
        
        if self.split_method == 'patient_wise':
            X_train = X_train[:, 1:]
            X_val = X_val[:, 1:]
        
        if self.category_info is None:
            self.category_info = np.zeros(X_train.shape[1])


        
        
        
        X_train_wis = X_train.copy()
        if self.winsorized:
            for i in range(X_train.shape[1]):
                X_train_wis[:,i,self.category_info==0] = Winsorizing(X_train[:,i,self.category_info==0])
                
        self.scaler = []
        for i in range(X_train.shape[1]):
            self.scaler.append(preprocessing.StandardScaler().fit(X_train_wis[:,i,self.category_info==0]))
            
        # add static feature
        # print(X_train_static.shape)
        # self.static_scaler = preprocessing.StandardScaler().fit(X_train_static[:, self.static_category_info==0])
        if X_train_static is not None:
            self.static_scaler = preprocessing.StandardScaler().fit(X_train_static[:, self.static_category_info==0])
            num_static_features = X_train_static.shape[-1]
        else:
            self.static_scaler = None
            self.static_category_info = None
            X_train_static = None
            X_val_static = None
            num_static_features = None
            
         
        _, _,train_loader = build_dataset_loader(X_train_wis, y_train, self.batch_size, 
                                                scaler=self.scaler, infinite=True,
                                                category_list=self.category_info,
                                                X_static = X_train_static,  
                                                static_category_info = self.static_category_info,
                                                static_scaler = self.static_scaler   
                                                  )
        
        scaled_train_X, scaled_train_static_X,train_loader_for_eval = build_dataset_loader(X_train_wis, y_train, self.batch_size,
                                                                    scaler=self.scaler, infinite=False,
                                                                    category_list=self.category_info,
                                                                    X_static = X_train_static, 
                                                                    static_category_info = self.static_category_info,
                                                                    static_scaler = self.static_scaler                         
                                                                                            )
        
        _, _, val_loader = build_dataset_loader(X_val, y_val, self.batch_size, 
                                                    scaler=self.scaler, infinite=False,
                                                    category_list=self.category_info,
                                                    X_static = X_val_static, 
                                                    static_category_info = self.static_category_info,
                                                    static_scaler = self.static_scaler
                                            )
        
        # Initialize the model using existing rules. 
        if self.init_rule_index is not None:
            if isinstance(self.init_rule_index, str):
                if self.rule_data is not None:
                    if self.init_rule_index == 'All':
                        init_data = self.rule_data
                    else:
                        init_rule_index = self.init_rule_index.split('_')
                        init_rule_index = [int(x) for x in init_rule_index]
                        # print(init_rule_index)
                        init_data = [self.rule_data[index] for index in init_rule_index]
                else:
                    raise ValueError
            else:
                init_data = self.init_rule_index
        else:
            init_data = None
        
        # Build network
        # n_variables, 
        #          n_rules,n_concepts, 
        #          n_classes, 
        #          category_info, 
        #          static_category_info, 
        #          n_static_variables = None, 
        #          n_visits = 4, 
        #          evolve_type = 'RNN', 
        #          epsilon_training = False
        net = Evolve_module(n_variables = len(self.category_info), 
                                           n_rules = self.n_rules, 
                                           n_concepts = 3, 
                                           n_classes = self.n_classes,
                                           category_info = self.category_info,
                                           static_category_info = self.static_category_info,
                                           n_static_variables = num_static_features,
                                           n_visits = self.n_visits,
                                           evolve_type = self.evolve_type,
                                           epsilon_training = self.epsilon_training) 
        
        if X_train_static is not None:
            temp_static_tensor = torch.from_numpy(scaled_train_static_X)
        else:
            temp_static_tensor = None
        net.reset_parameters(torch.from_numpy(scaled_train_X),temp_static_tensor,init_data)
        net.to(device)
        
        self.initial_weights = copy.deepcopy(net.state_dict())
        # Build loss function and optimizer

        if self.weighted_loss is not None:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.weighted_loss, dtype=dtype, device=device))
            # criterion = utils.FocalLoss(gamma=0.1)
        else:
            criterion = nn.CrossEntropyLoss()
        
        weight_decay = 1e-4
        parameters = utils.add_weight_decay(net, weight_decay)
        optimizer = optim.Adam(parameters, lr=self.learning_rate)
        
        
        train_aucpr_list = []
        val_aucpr_list = []
            
        start_epsilon = 0.99
        
        patience = 0
        best_value = 0
        best_step = 0
        best_net = copy.deepcopy(net.state_dict())
        best_epsilon = start_epsilon
        delay_step = -1
        for global_step, (inputs, labels,static_features) in enumerate(train_loader, 0):
        # for global_step, (inputs, labels) in enumerate(train_loader, 0):
            labels = labels.type(torch.long)
            # Optimize
            optimizer.zero_grad()
            
            # Exponentially reduce the epsilon value
            epsilon = max(start_epsilon*(0.999**((global_step-delay_step)//2)), self.min_epsilon)
            if global_step>100:
                for g in optimizer.param_groups:
                    g['lr'] = self.learning_rate*0.5
            
            
            inputs_list = []
            for i in range(inputs.shape[-1]):
                inputs_list.append(inputs[:,:,i])
            # out,attention_masks_continuous_list,attention_masks_categorical_list,connection_list,variable_contrib_list,rule_contrib_list
            out,connection_list,attention_masks_continuous_list, attention_masks_categorical_list,static_attention_masks_continuous, static_attention_masks_categorical = net(inputs_list, epsilon,static_features) # Forward pass
            

            regu_1, regu_2 = self._regularization_calculation(connection_list,
                                                        attention_masks_continuous_list,
                                                        attention_masks_categorical_list,
                                                        static_attention_masks_continuous, 
                                                        static_attention_masks_categorical)
            # regu_1 = 0
            # regu_2 = 0
            cross_entropy =  criterion(out, labels)
            loss = cross_entropy+regu_1+regu_2
            loss.backward()        
            
            for param in net.parameters():
                if param.grad is not None:
                    param.grad[torch.isnan(param.grad)] = 0
            
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
            

            if (global_step+1) % self.report_freq == 0:
                # if self.epsilon_training:
                    # print(net.layer2.epsilon.detach().numpy())
                    # print(net.layer3.epsilon.detach().numpy())
                    
                _, _, _, _, f1_train, auc_train, aucpr_train, _, _ = self._model_testing(net, train_loader_for_eval, epsilon)
                _, _, _, _, f1_val, auc_val, aucpr_val, _, _ = self._model_testing(net, val_loader, epsilon)
                train_aucpr_list.append(aucpr_train)
                val_aucpr_list.append(aucpr_val) 
                
                if epsilon == self.min_epsilon:
                    if auc_val>best_value:
                        best_value = auc_val
                        best_step = global_step
                        best_net = copy.deepcopy(net.state_dict())
                        patience = 0
        
                    else:
                        patience += 1
        
                    if patience > self.patience_step//self.report_freq:
                        break
                
                if self.verbose==2:
                    print(epsilon)
                    print(f'Step {global_step}, loss :{loss:.4f}, train_auc: {auc_train:.3f}, train_aucpr: {aucpr_train:.3f}, train_f1: {f1_train:.3f}, val_auc: {auc_val:.3f}, val_aucpr: {aucpr_val:.3f}, val_f1: {f1_val:.3f}.')
            if global_step > self.max_steps:
                break

        if self.verbose>=1:
            print(f'The best value is {best_value:.2f} at step {best_step} with epsilon {best_epsilon:.3f}.')
        
        net.load_state_dict(best_net)
        self.estimator = net
        
        self.classes_ = unique_labels(y_train)
        # self.X_ = X
        # self.y_ = y
        self.epsilon = best_epsilon

        return self
    

    def predict(self, X, X_test_static = None):
        """ Predict the class of the given data samples.
        """
        if self.split_method == 'patient_wise':
            X = X[:, 1:]
            
        # Check is fit had been called
        check_is_fitted(self)

        
        _, _,test_loader = build_dataset_loader(X, batch_size=self.batch_size, scaler=self.scaler,
                                           infinite=False, category_list=self.category_info,
                                                    X_static = X_test_static, 
                                                    static_category_info = self.static_category_info,
                                                    static_scaler = self.static_scaler)
            
        pred_list = []
        for i, (inputs,static) in enumerate(test_loader, 0):
        # for i, (inputs) in enumerate(test_loader, 0):
            inputs_list = []
            for i in range(inputs.shape[-1]):
                inputs_list.append(inputs[:,:,i])
            x,_,_,_,_,_= self.estimator(inputs_list, self.epsilon,static) #out,attention_masks_continuous_list,attention_masks_categorical_list,connection_list,rule_contrib_list
            pred = torch.argmax(x, dim=1).detach().cpu().numpy()
            pred_list.append(pred)
            
        pred_list = np.concatenate(pred_list, axis=0)

        return pred_list
    
    
    def predict_proba(self, X,X_test_static):
        """ Predict the probabilites of belonging to individaul classes of the given data samples.
        """
        if self.split_method == 'patient_wise':
            X = X[:, 1:]
            
        check_is_fitted(self)

        # Input validation
        _, _,test_loader = build_dataset_loader(X, batch_size=self.batch_size, scaler=self.scaler,
                                           infinite=False, category_list=self.category_info,
                                                    X_static = X_test_static, 
                                                    static_category_info = self.static_category_info,
                                                    static_scaler = self.static_scaler)
        
            
        prob_list = []
        # static = None
        for i, (inputs,static) in enumerate(test_loader, 0):
        # for i, (inputs) in enumerate(test_loader, 0):
            inputs_list = []
            for i in range(inputs.shape[-1]):
                inputs_list.append(inputs[:,:,i])
            # out,attention_masks_continuous_list,attention_masks_categorical_list,prev_connection_list,last_connection_list,variable_contrib_list,rule_contrib_list
            x,_,_,_,_,_= self.estimator(inputs_list, self.epsilon,static)

            prob = F.softmax(x, dim=1)
            prob_list.append(prob.detach().cpu().numpy())        
            
        prob_list = np.concatenate(prob_list, axis=0)
        prob_list = np.round(prob_list, 3)
        
        return prob_list
    
    def evaluate(self, X, X_test_static,y_test):
        """ Predict the probabilites of belonging to individaul classes of the given data samples.
        """
        if self.split_method == 'patient_wise':
            X = X[:, 1:]
            
        check_is_fitted(self)

        # Input validation
        _, _,test_loader = build_dataset_loader(X, y_test, batch_size=self.batch_size, scaler=self.scaler,
                                              infinite=False, category_list=self.category_info,
                                                    X_static = X_test_static, 
                                                    static_category_info = self.static_category_info,
                                                    static_scaler = self.static_scaler)

        
            
        prob_list = []
        label_list = []
        for i, (inputs, labels,static) in enumerate(test_loader, 0):
            inputs_list = []
            for i in range(inputs.shape[-1]):
                inputs_list.append(inputs[:,:,i])
            x,_,_,_,_,_= self.estimator(inputs_list, self.epsilon,static)

            prob = F.softmax(x, dim=1)
            label_list.append(labels)
            prob_list.append(prob.detach())     
            
        label_list = torch.cat(label_list, dim=0)
        prob_list = torch.cat(prob_list, dim=0)
            
        labels = label_list.cpu().numpy()
        probs = prob_list.cpu().numpy()
        probs = np.round(probs, 3)
        
        aucpr = sklearn.metrics.average_precision_score(labels, probs[:,1])
        return aucpr
        
        
        
        
        
        
        
        
        return prob_list
    
    
    def _regularization_over_feature_continuous(self,attention_mask_continous,category_info,normtype = 1):
        regu = 0
        n_var = np.sum(category_info == 0)
        n_rule = attention_mask_continous.shape[-1]
        start_idx = 0
        for feature_idx in range(n_var):
                feature = attention_mask_continous[feature_idx, :,:]
                feature_norm = torch.norm(feature.view(-1),p = normtype)  # L1-norm
                regu += feature_norm
        return regu
    
    def _regularization_over_feature_categorical(self,attention_mask_categorical,category_info):
        regu = 0
        n_var = np.sum(category_info > 0)
        n_category = category_info[category_info>0]
        start_idx = 0
        for feature_idx in range(n_var):
            feature = attention_mask_categorical[start_idx:start_idx+n_category[feature_idx], :]
            feature_norm = torch.norm(feature)  # L1-norm
            start_idx += n_category[feature_idx]
            regu += feature_norm
        return regu
        
        
    
    
    
    
    
    
    
    def _regularization_calculation(self, connection_mask, attention_mask_continous, attention_mask_categorical,static_attention_masks_continuous, static_attention_masks_categorical):
        
        category_info = self.category_info
        static_category_info = self.static_category_info
        
        # regulization for continuous variables 
        n_continous_variables = np.sum(category_info==0)
        n_category_variables = np.sum(category_info>0)
        # attention_regu_continous = torch.norm(attention_mask_continous.view(-1), 1)
        
        attention_regu_continous = self._regularization_over_feature_continuous(attention_mask_continous,category_info)

        if n_category_variables>0:
            # attention_regu_categorical = torch.norm(attention_mask_categorical.view(-1), 1)
            attention_regu_categorical = self._regularization_over_feature_categorical(attention_mask_categorical,category_info)
            attention_regu = attention_regu_continous + attention_regu_categorical
        else:
            attention_regu = attention_regu_continous
        
        # add static 
        n_continous_variables_static = np.sum(static_category_info==0)
        n_category_variables_static = np.sum(static_category_info>0)

        # static_attention_regu_continous = torch.norm(static_attention_masks_continuous.view(-1), 1)
        static_attention_regu_continous = self._regularization_over_feature_continuous(static_attention_masks_continuous,static_category_info)
        if n_category_variables_static>0:
            # static_attention_regu_categorical = torch.norm(static_attention_masks_categorical.view(-1), 1)
            static_attention_regu_categorical  = self._regularization_over_feature_categorical(static_attention_masks_categorical,static_category_info)
            static_attention_regu = static_attention_regu_continous + static_attention_regu_categorical
        else:
            static_attention_regu = static_attention_regu_continous
        
        # final Sparse regularization
        connection_regu = torch.norm(connection_mask.view(-1), 1)
        regu_1 = self.sparse_regu * (static_attention_regu) + 1e-5*attention_regu
        
        # Build the rule matrix 
        all_cate_info = np.concatenate([category_info,static_category_info]).astype(np.int8)
        
        
        # attention_continous = attention_mask_continous.view([-1, self.n_rules])
        # attention_categorical = attention_mask_categorical.view([-1, self.n_rules])
        
        # static_attention_continous = static_attention_masks_continuous.view([-1, self.n_rules])
        # static_attention_categorical = static_attention_masks_categorical.view([-1, self.n_rules])
        
        # all_attention_continous = torch.cat([attention_continous,static_attention_continous],axis = 0)
        # all_attention_categorical = torch.cat([attention_categorical,static_attention_categorical],axis = 0)
        
        all_attention_continous = torch.cat([attention_mask_continous,static_attention_masks_continuous],axis = 0)
        all_attention_categorical = torch.cat([attention_mask_categorical,static_attention_masks_categorical],axis = 0)
        
        num_continous_x = np.sum(all_cate_info==0)
        num_cat_x = np.sum(all_cate_info>0)
        # print('all_attention_continous:',all_attention_continous.shape)
        # print('connection_mask:',torch.stack([connection_mask[0:num_continous_x, :]]*3, axis=1).shape)
        mat = all_attention_continous * torch.stack([connection_mask[0:num_continous_x, :]]*3, axis=1)
        mat = mat.reshape(-1, self.n_rules)
        
        if num_cat_x>0:
            n_category = all_cate_info[all_cate_info>0]
            attention_category_list = torch.split(all_attention_categorical, list(n_category))
            mat_category_list = []
            for i in range(num_cat_x):
                temp = all_attention_categorical[i]*torch.stack([connection_mask[num_continous_x+i]]*n_category[i], axis=0)
                mat_category_list.append(temp)
            
            mat = torch.cat([mat, torch.cat(mat_category_list, dim=0)],  dim=0)
        
        # Correlation regularization
        regu_2 = 0
        for i in range(self.n_rules):
            for j in range(i, self.n_rules):
                regu_2 += torch.sum(mat[:, i]*mat[:, j])/(
                        torch.norm(mat[:, i], 2)*torch.norm(mat[:, j], 2)+0.0001)
        regu_2 = self.corr_regu * regu_2
        return regu_1, regu_2
    
    
    def _model_testing(self, net, test_loader, epsilon):
        """ Model test.
        
        Parameters
        ----------
        net: A Net object. The network with the best validation performance
        test_loader: finite data_loader for evaluation.
        epsilon: A float. The current epsilon value.
        
        Returns
        -------
        Evaluation metrics includeing accuracy, sensitivity, specificity, precision, f1-score, auc, and aucpr.
        """
        pred_list = []
        label_list = []
        prob_list = []
        for i, (inputs, labels,static) in enumerate(test_loader, 0):
        # for i, (inputs, labels) in enumerate(test_loader, 0):    
            inputs_list = []
            for i in range(inputs.shape[-1]):
                inputs_list.append(inputs[:,:,i])
            
            #inputs, labels = data
            labels = labels.type(torch.long)
            x,_,_,_,_,_ = net(inputs_list, epsilon,static) # x, _,_,prev_connection_list,last_connection_list,variable_contrib_list,rule_contrib_list
            
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
            
        pred = pred_list.cpu().numpy()
        labels = label_list.cpu().numpy()
        probs = prob_list.cpu().numpy()
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
