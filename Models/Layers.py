"""
Author: Yufeng Zhang

"""


import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
import numpy as np

dtype = torch.float32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes)
    labels =y[labels]
    return labels
      
class InputLayer(nn.Module):
    """
    Encode the input variables into concepts.
    For continuous variables, they will be encoded using fuzzified concetps.
    For categorical variables, they will be encoded by one-hoting coding.
    """
    def __init__(self, category_info):
        
        super(InputLayer, self).__init__()
        self.category_info = category_info
        self.n_continuous_variables = np.sum(category_info==0)
        self.n_categorical_variables = np.sum(category_info>0)
        self.weight = Parameter(torch.Tensor(4, self.n_continuous_variables))
        
    
    def f(self, value):
        epsilon = self.epsilon
        i1 = value>=0
        i2 = value<0
        
        out = torch.zeros_like(value)
        out[i1] = value[i1] + epsilon*torch.log(torch.exp(-value[i1]/epsilon)+1)
        out[i2] = epsilon*torch.log(1+torch.exp(value[i2]/epsilon))
        return out

    
    def forward(self, variables, epsilon):
        """
        Parameters
        ----------
        variables: torch.Tensor. the original feature matrix with a shape of (batch_size, n_variables)
        
        Returns
        -------
        x_continuous: torch.Tensor. Encoded continuous variables in a shape of (batch_size, n_continuous_variables, n_concepts), 
                where n_concepts=3.
        x_categorical_list:  A list of encoded categorical variables. 
        """
        input_continuous = variables[:, self.category_info==0]
        input_categorical = variables[:, self.category_info>0]
        
        category_levels = self.category_info[self.category_info>0]

        self.epsilon = epsilon
        # Calculate a1, a2, b1, b2 lists, which define membership functions for individual continuous variables
        self.a2_list = self.weight[0,:] - 0.1 - self.weight[1,:].pow(2)
        self.b1_list = self.weight[0,:] + 0.1 + self.weight[1,:].pow(2)
        self.a1_list = self.a2_list - 0.1 - self.weight[2,:].pow(2)
        self.b2_list = self.b1_list + 0.1 + self.weight[3,:].pow(2)
        
        batch_size = variables.shape[0]
        a1_batch = torch.unsqueeze(self.a1_list, 0).repeat(batch_size, 1)
        a2_batch = torch.unsqueeze(self.a2_list, 0).repeat(batch_size, 1)
        b1_batch = torch.unsqueeze(self.b1_list, 0).repeat(batch_size, 1)
        b2_batch = torch.unsqueeze(self.b2_list, 0).repeat(batch_size, 1)
        
        # Calculate membership values of indiviudal continuous variables to low, medium, high concepts, respectively
        lx =  self.f((a2_batch-input_continuous)/(a2_batch-a1_batch)) - self.f((a1_batch-input_continuous)/(a2_batch-a1_batch))
        mx = self.f((input_continuous-a1_batch)/(a2_batch-a1_batch)) - self.f((input_continuous-a2_batch)/(a2_batch-a1_batch)) \
            + self.f((b2_batch-input_continuous)/(b2_batch-b1_batch)) - self.f((b1_batch-input_continuous)/(b2_batch-b1_batch)) - 1
        hx = self.f((input_continuous-b1_batch)/(b2_batch-b1_batch)) - self.f((input_continuous-b2_batch)/(b2_batch-b1_batch))
        
        x_continuous = torch.stack([lx, mx, hx], axis=-1)
        x_continuous = F.relu(x_continuous)
        x_categorical_list = []

        # Categorical variables are encoded using regular one-hot embedding.
        for i in range(input_categorical.shape[1]):
            x = input_categorical[:,i]
            x = x.type(torch.long)
            out = one_hot_embedding(x, int(category_levels[i]))
            out = out.type(dtype)
            x_categorical_list.append(out)
        return x_continuous, x_categorical_list
            
        
    def reset_parameters(self, m_list, s_list):
        # Initialize the parameters of membership function using the mean and standard deviation of the training data. 
        weight = torch.stack([m_list, torch.sqrt(s_list), torch.sqrt(s_list), torch.sqrt(s_list)], dim=0)
        self.weight.data = weight

class RuleLayer_for_concepts(nn.Module):
    def __init__(self, n_concepts, n_rules, category_info):
        super(RuleLayer_for_concepts, self).__init__()
        self.n_concepts = n_concepts
        self.n_variables = len(category_info)
        self.n_continuous_variables = np.sum(category_info==0)
        self.n_categorical_variables = np.sum(category_info>0)
        self.category_levels = category_info[category_info>0]
        self.n_rules = n_rules

        
        # Initiate parameters
        self.attention_continuous = Parameter(torch.Tensor(self.n_continuous_variables, n_concepts, n_rules))
        self.attention_categorical = Parameter(torch.Tensor(int(np.sum(self.category_levels)), n_rules))
        

    
    def forward(self, x_continuous, x_categorical_list):
        """
        

        Parameters
        ----------
        x_continuous : torch.Tensor. Encoded continuous variables in a shape of (n_concepts, n_variables), where n_concepts=3,
            which is generated from the encoding layer.
        x_categorical_list : List. A list of encoded categorical variables from the encoding layer.
        epsilon : Float. 

        Returns
        -------
        out : torch.Tensor with a shape of (batch_size, n_rules). Firing strength of individual rules.
        connection_mask : torch.Tensor with a shape of (n_variables, n_rules). Connection matrix.
        attention_masks_continuous : torch.Tensor with a shape of (n_continuous_variables, n_concepts, n_rules). 
                Attention matrix for continuous variables.
        attention_masks_categorical : torch.Tensor with a shape of (n_categorical_variables, n_rules). 
                Attention matrix for categorical variables.
        varible_contrib : torch.Tensor with a shape of (batch_size, n_variables, n_rules). Firing strength of 
                individual variables to individual rules. This information will be used for the final rule
                clustering and summarization.

        """
        n_samples = x_continuous.shape[0]
        attention_masks_continuous = (torch.tanh(self.attention_continuous) + 1)/2
        attention_masks_categorical = (torch.tanh(self.attention_categorical) + 1)/2
        
        out = []
        
        x_continuous_stack = torch.unsqueeze(x_continuous, -1).repeat(1, 1, 1, self.n_rules)
        amask_batch = torch.unsqueeze(attention_masks_continuous, 0).repeat(n_samples, 1, 1, 1)

        hidden = torch.mul(x_continuous_stack, amask_batch)
        hidden = torch.sum(hidden, dim=-2)
            
        out_category = []
        for i in range(self.n_rules):
            # For category variables
            if self.n_categorical_variables>0:
                hidden_category= []
                category_mask_list = torch.split(attention_masks_categorical, list(self.category_levels))
                for j in range(self.n_categorical_variables):
                    hidden_category.append(torch.matmul(x_categorical_list[j].to(device), category_mask_list[j][:, i].to(device)))
                hidden_category = torch.stack(hidden_category, dim=1).to(device)
                out_category.append(hidden_category)
        
        if len(out_category)>0:
            hidden = torch.cat([hidden, torch.stack(out_category, axis=-1)], axis=1)
            
        variable_contrib = hidden
        hidden = 1-F.relu(1-hidden) + 0.1**5
        
        return out, attention_masks_continuous, attention_masks_categorical, hidden, variable_contrib
    
    
    def reset_parameters(self):       
        value1 = 0
        value2 = 0
        nn.init.uniform_(self.attention_continuous, a=value1-1, b=value1)
        nn.init.uniform_(self.attention_categorical, a=value1-1, b=value1)
        

class RuleLayer_for_variables(nn.Module):
    """
    Calculate rules.
    """
    def __init__(self,  category_info, epsilon_training=False):
        super(RuleLayer_for_variables, self).__init__()
        self.n_variables = len(category_info)
        self.epsilon_training = epsilon_training
        
        
        if self.epsilon_training:
            self.epsilon = Parameter(torch.Tensor([0.1]))
    
    
    def forward(self,n_samples, hidden, connection_mask, epsilon):
        """

        Returns
        -------
        out : torch.Tensor with a shape of (batch_size, n_rules). Firing strength of individual rules.
        connection_mask : torch.Tensor with a shape of (n_variables, n_rules). Connection matrix.

        """
        if self.epsilon_training:
            epsilon = max(0.1, 1/(self.epsilon.pow(2)+1)) #.pow(0.5)
        
        
        out = []

        temp = hidden.pow(connection_mask*(epsilon-1)/epsilon)   
        # print('layer for variables 1')
        # print(temp[0,0,1])
        temp = torch.sum(temp, dim=1) - (self.n_variables-1)
        # temp = torch.sum(temp, dim=1)/ torch.sqrt(torch.tensor(self.n_variables, dtype=torch.float))
        #         print('layer for variables 1')
        # print(temp[0,0,1])
        out = temp.pow(epsilon/(epsilon-1))
        
        if torch.sum(torch.isnan(out))>0 or torch.sum(torch.isinf(out))>0:
            print('rule layer error')
        return out
    


class InferenceLayerPos(nn.Module):
    """
    Calculate rules.
    With InferenceLayerPos, all rules encoded in the network will only contribute the positive class.
    This will only be used in a binary classification task.
    """
    def __init__(self, n_rules, n_classes, epsilon_training=False):
        super(InferenceLayerPos, self).__init__()
        self.n_rules = n_rules
        self.n_classes = n_classes
        self.epsilon_training = epsilon_training
        
        # Initiate parameters
        self.weight = Parameter(torch.Tensor(n_rules))
        # Use a fixed bias
        self.bias = torch.Tensor([-2]) #-1, -2
        if self.epsilon_training:
            self.epsilon = Parameter(torch.Tensor([0.1]))

    def forward(self, x, epsilon):
        """
        Calculate the firing strength of individual rules. 

        Parameters
        ----------
        x : torch.Tensor with a shape of (batch_size, n_rules). Firing strength of individual rules from the rule layer.
        epsilon : Float.

        Returns
        -------
        out : torch.Tensor with a shape of (batch_size, 2).

        """
        n_samples = x.shape[0]
        if self.epsilon_training:
            epsilon = max(0.1, 1/(self.epsilon.pow(2)+1)) #.pow(0.5)
            
        out = torch.mul(x, torch.stack([self.weight.pow(2).pow(0.5)]*n_samples, axis=0))  
        rule_contrib = out
        temp = torch.sum(out.pow(1/epsilon), axis=-1)

        out = torch.stack([torch.zeros(temp.shape).to(device), temp + self.bias.to(device)], dim=-1)
        return out, rule_contrib
    
    def reset_parameters(self):
        nn.init.uniform_(self.weight, a=0.1, b=0.1) #1
        if self.epsilon_training:
            self.epsilon.data = torch.Tensor([1])
