# -*- coding: utf-8 -*-
"""
@author: Yufeng Zhang
"""
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
from Layers import *




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Evolve_module(nn.Module):
    def __init__(self, n_variables, 
                 n_rules,n_concepts, 
                 n_classes, 
                 category_info, 
                 static_category_info, 
                 n_static_variables = None, 
                 n_visits = 4, 
                 evolve_type = 'RNN', 
                 epsilon_training = False):
        super(Evolve_module,self).__init__()
        
        self.category_info = category_info
        self.n_visits = n_visits
        self.epsilon_training = epsilon_training
        
        self.layer_encode = InputLayer(category_info)        
        self.layer_concepts = RuleLayer_for_concepts(n_concepts, n_rules, category_info)
        self.layer_inference = InferenceLayerPos(n_rules, n_classes, epsilon_training=epsilon_training)
        self.evolve_type = evolve_type
        if self.evolve_type == 'RNN':
            self.RNN_update = RNN_update_cell(n_variables,n_rules)
        elif self.evolve_type == 'GRU':
            self.RNN_update = GRU_update_cell(n_variables,n_rules)
        elif self.evolve_type == 'LSTM':
            self.RNN_update = LSTM_update_cell(n_variables,n_rules)
            self.cell_state = Parameter(torch.Tensor(n_variables,n_rules))

        
        self.connect_init = Parameter(torch.Tensor(n_variables,n_rules))
        self.reset_connect(self.connect_init)
        
        
        
        
        
        ##### add static 
        self.static_category_info = static_category_info
        if n_static_variables is not None:
            self.static_connect_init = Parameter(torch.Tensor(n_static_variables,n_rules))
            self.reset_connect(self.static_connect_init)
    
            self.static_layer_encode = InputLayer(static_category_info) 
            self.static_layer_concepts = RuleLayer_for_concepts(n_concepts, n_rules, static_category_info)
            all_category_info = np.concatenate([category_info,static_category_info],axis = 0)
            self.layer_variables = RuleLayer_for_variables(all_category_info, epsilon_training=epsilon_training)
        else:
            self.layer_variables = RuleLayer_for_variables(category_info, epsilon_training=epsilon_training)

    
    def forward(self,inputs_list,epsilon,static_features = None):

        connection = self.connect_init
        connection_mask = (torch.tanh(connection) + 1)/2 

        
        ##### add static  outside the RNN framework
        if static_features is not None:
            static_connection = self.static_connect_init
            static_connection_mask = (torch.tanh(static_connection) + 1)/2
            x_static_continuous, x_static_category_list = self.static_layer_encode(static_features, epsilon)
            n_samples, static_attention_masks_continuous, static_attention_masks_categorical, static_hidden, static_variable_contrib = self.static_layer_concepts(x_static_continuous, x_static_category_list)

            self.static_attention_masks_continuous = static_attention_masks_continuous
            self.static_attention_masks_categorical = static_attention_masks_categorical
        

        if self.evolve_type == 'LSTM':
            cell_state = self.cell_state
            
        for l,inputs in enumerate(inputs_list):
            x_continuous, x_category_list = self.layer_encode(inputs, epsilon)
            
            n_samples, attention_masks_continuous, attention_masks_categorical, hidden, variable_contrib = self.layer_concepts(x_continuous, x_category_list)
            
            
            if self.evolve_type == 'LSTM':
                connection_mask,cell_state = self.RNN_update(n_samples, hidden, connection_mask) # connection_mask: (n_samples, n_variables,n_rules) ,cell_state
            else:
                connection_mask = self.RNN_update(n_samples, hidden,connection_mask) 

            # x2 = self.layer_variables(n_samples,hidden,connection_mask, epsilon)

            if l == self.n_visits-1:
                if static_features is not None:
                    all_hidden = torch.cat([hidden, static_hidden],axis = 1)
                    # print('combined hidden')
                    # print(all_hidden[:4,1,0])
                    all_connection_mask = torch.cat([connection_mask,static_connection_mask],axis = 0)
                    x2 = self.layer_variables(n_samples,all_hidden,all_connection_mask, epsilon)
                    # print('combined x2')
                    # print(x2[:4,1])
                    x3, rule_contrib = self.layer_inference(x2, epsilon)
                    self.connection = all_connection_mask

                    
                else:
                    x2 = self.layer_variables(n_samples,hidden,connection_mask, epsilon)
                    x3, rule_contrib = self.layer_inference(x2, epsilon)
                    self.connection = connection_mask
                # x3, rule_contrib = self.layer_inference(x2, epsilon)
                self.rule_contrib = rule_contrib
                self.variable_contrib = variable_contrib
                self.attention_masks_continuous = attention_masks_continuous
                self.attention_masks_categorical = attention_masks_categorical
            # all_connection_mask =  connection_mask
            # static_attention_masks_continuous = 0 
            # static_attention_masks_categorical = 0
            
            
        # return x3,connection_mask,attention_masks_continuous, attention_masks_categorical
        return x3,all_connection_mask,attention_masks_continuous, attention_masks_categorical,static_attention_masks_continuous, static_attention_masks_categorical

    def reset_connect(self,t):
        nn.init.uniform_(t, a = 0, b= 2)
    def reset_cell_state(self,t):
        nn.init.kaiming_uniform_(t,nonlinearity = 'relu')
    def reset_parameters(self, train_features,static_features,init_data = None):
        features_continuous = train_features[:,0,self.category_info==0]
        m_list = torch.mean(features_continuous, dim=0) 
        s_list = torch.std(features_continuous, dim=0) * 2
        self.layer_encode.reset_parameters(m_list, s_list)
        self.layer_concepts.reset_parameters()
        self.layer_inference.reset_parameters()
        
        # add static
        if static_features is not None:
            staic_features_continuous = static_features[:,self.static_category_info==0]
            m_list = torch.mean(staic_features_continuous, dim=0) 
            s_list = torch.std(staic_features_continuous, dim=0)/2
            self.static_layer_encode.reset_parameters(m_list, s_list)
            self.static_layer_concepts.reset_parameters()
        
        if init_data is not None:
            attention_continuous = self.layer_concepts.attention_continuous.data
            attention_categorical = self.layer_concepts.attention_categorical.data
            connection = self.connect_init.data
            weight = self.layer_inference.weight.data
            
            attention_continuous, attention_categorical, connection, out = self._initiate_weights_with_given_rules(
                attention_continuous, attention_categorical, connection, weight, init_data, init_value=1.0)
            
            self.layer_concepts.attention_continuous.data = attention_continuous
            self.layer_concepts.attention_categorical.data = attention_categorical
            self.connect_init.data = connection
            self.layer_inference.weight.data = out
            
        
        
    def _initiate_weights_with_given_rules(self,attention_continuous, attention_categorical, connection, weight, init_data, init_value=1.0):
        
        n_continuous_variables = np.sum(self.category_info==0)
        category_levels = self.category_info[self.category_info>0]
        
        # Build the delta vector from category_info
        delta = []
        d_continuous = 0
        d_categorical = 0
        for i in range(len(self.category_info)):
            if self.category_info[i] == 0:
                d_categorical += 1
                delta.append(d_continuous)
            else:
                d_continuous += 1
                delta.append(d_categorical)
              
        for rule_index, rule in enumerate(init_data):
            temp_attention_continuous = -torch.ones_like(attention_continuous[:,:,0])
            temp_attention_categorical = -torch.ones_like(attention_categorical[:,0])
            temp_connection = -torch.ones_like(connection[:,0])
            
            for concept in rule['Relation']:
                variable_index = concept[0] - delta[concept[0]]
                concept_index = concept[1]
        
                if self.category_info[concept[0]] == 0:
                    temp_attention_continuous[variable_index, concept_index] = init_value
                    temp_connection[variable_index] = init_value
                else:
                    temp_attention_categorical[np.sum(
                        category_levels[:variable_index])+concept_index] = init_value
                    temp_connection[n_continuous_variables+variable_index] = init_value
            attention_continuous[:,:,rule_index] = temp_attention_continuous
            attention_categorical[:,rule_index] = temp_attention_categorical
            connection[:,rule_index] = temp_connection

            weight[rule_index] = rule['Out_weight']
           
        return attention_continuous, attention_categorical, connection, weight
        
        
        
    
class RNN_update_cell(nn.Module):
    def __init__(self, n_variables,n_rules):
        super().__init__()
        self.W = Parameter(torch.Tensor(n_variables,n_variables))
        self.U = Parameter(torch.Tensor(n_variables,n_variables))
        self.bias = Parameter(torch.zeros(n_variables,n_rules))
        
        self.reset_param(self.W)
        self.reset_param(self.U)
        self.reset_bias(self.bias)
        
    def reset_param(self,w):
        nn.init.kaiming_uniform_(w,nonlinearity = 'relu')
        # nn.init.uniform_(w, a=0, b=1)
    
    def reset_bias(self,w):
        nn.init.uniform_(w, a=0, b=1)
        # nn.init.uniform_(w, a=-0.1, b=0.1)
            
    def forward(self, n_samples, connect_contrib, prev_connect):
        """  
        Args:
            connect_contrib: n_variables * n_rules (X)
            prev_connect: n_variables * n_rules (H)
            
        Return:
            connect: n_variables * n_rules
        """
        # if not personalized:
        #     # prev_connect = torch.unsqueeze(prev_connect, 0).repeat(n_samples, 1, 1)
        #     # connect_contrib = torch.mean(connect_contrib, 0, False)
        #     # print('The shape of connect_contrib:',connect_contrib.shape)
        connect = self.W.matmul(connect_contrib) + self.U.matmul(prev_connect) + self.bias
        connect =  (torch.tanh(torch.mean(connect, 0, False))+1)/2
        # else:
        # connect = (torch.tanh(self.W.matmul(connect_contrib) + self.U.matmul(prev_connect) + self.bias)+1)/2
        # print('The shape of connect:',connect.shape)

        return connect


class GRU_update_cell(nn.Module):
    def __init__(self, n_variables,n_rules):
        super().__init__()
        self.update = mat_GRU_gate(n_variables,n_rules,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(n_variables,n_rules,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(n_variables,n_rules,
                                   torch.nn.Tanh())
            
    def forward(self, n_samples, connect_contrib, prev_connect):
        """  
        Args:
            connect_contrib: n_variables * n_rules (X)
            prev_connect: n_variables * n_rules (H)
            
        Return:
            connect: n_variables * n_rules
        """
        # connect = self.W.matmul(connect_contrib) + self.U.matmul(prev_connect) + self.bias
        # connect =  (torch.tanh(torch.mean(connect, 0, False))+1)/2
        

        update = self.update(connect_contrib,prev_connect)
        reset = self.reset(connect_contrib,prev_connect)

        h_cap = reset * prev_connect
        h_cap = self.htilda(connect_contrib, h_cap)

        new_Q = (1 - update) * prev_connect + update * h_cap
        connect =  (torch.tanh(torch.mean(new_Q, 0, False))+1)/2
        # connect =  torch.mean((new_Q +1)/2,0, False)
        return connect
    
class LSTM_update_cell(nn.Module):
    def __init__(self, n_variables,n_rules):
        super().__init__()
        self.update = mat_GRU_gate(n_variables,n_rules,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(n_variables,n_rules,
                                   torch.nn.Sigmoid())
        
        self.output = mat_GRU_gate(n_variables,n_rules,
                                   torch.nn.Sigmoid())
        self.htilda = mat_GRU_gate(n_variables,n_rules,
                                   torch.nn.Tanh())
            
    # def forward(self, n_samples, connect_contrib, prev_connect,cell_state):
    def forward(self, n_samples, connect_contrib, prev_connect):
        """  
        Args:
            connect_contrib: n_variables * n_rules (X)
            prev_connect: n_variables * n_rules (H)
            
        Return:
            connect: n_variables * n_rules
        """


        update = self.update(connect_contrib,prev_connect) # (0,1)
        reset = self.reset(connect_contrib,prev_connect) # (0,1)
        ot = self.output(connect_contrib,prev_connect) # (0,1)
        cell_tilda  =  self.htilda(connect_contrib,prev_connect) # (-1,1)
        
        cell = torch.tanh(update * prev_connect + reset * cell_tilda) # (-1,1)

        new_Q = ot * cell
        connect =  (torch.tanh(torch.mean(new_Q, 0, False))+1)/2
        return connect,cell

    
class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))
        self.reset_param(self.bias)

    def reset_param(self,t):
        nn.init.kaiming_uniform_(t,nonlinearity = 'relu')
    
    def reset_bias(self,w):
        nn.init.uniform_(w, a=0, b=1)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out