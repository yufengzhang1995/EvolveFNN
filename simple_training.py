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

n_bucket = 4
hold_off = 180
observation = 360
visit_type = 'bucket'
evolve_type = 'RNN'

dataset_root = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Code/Yufeng/FNN_evolve/dataset'
ffile_root = os.path.join(dataset_root,f'{visit_type}_based/UMHS_lab_vital_{n_bucket}_{visit_type}s_h_{hold_off}_o_{observation}.p')
dataset = pickle.load(open(ffile_root,'rb'))

out_root = 'models'
unique_id = f'Dynamic_FNN_{n_bucket}_{visit_type}_hold_off_{hold_off}_{evolve_type}'

exp_save_path = os.path.join(out_root, unique_id)
if not os.path.isdir(out_root):
    os.mkdir(out_root)
if not os.path.isdir(exp_save_path):
    os.mkdir(exp_save_path)
print('######################################')
print('Experiment ID:', unique_id)
print('######################################')

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
static_category_info = static_category_info.astype(np.int32)

time_varying_features = data[:,0:num_time_varying_features*n_bucket]
time_invariant_features = data[:,-num_time_invariant_features:]
max_steps = 800
random_state = 1234

ss_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
X, y, X_test,y_test = utils.split_dataset(ss_train_test, data, labels, index=0)
X_train, y_train, X_val, y_val = utils.split_dataset(ss_train_test, X, y, index=0)
X_train = utils.fill_in_missing_value(X_train,X_train)
X_val = utils.fill_in_missing_value(X_val,X_val)
X_train_variant,X_train_invariant = utils.handle_data(X_train,num_time_invariant_features,n_bucket)
X_val_variant,X_val_invariant = utils.handle_data(X_val,num_time_invariant_features,n_bucket)

X_test = utils.fill_in_missing_value(X_test,X_test)
X_test_variant,X_test_invariant = utils.handle_data(X_test,num_time_invariant_features,n_bucket)
print('The shape of training time-varying data:',X_train_variant.shape)
print('The shape of validation time-varying data:',X_val_variant.shape)
print('The shape of testing time-varying data:',X_test_variant.shape)

device = 'cpu'
max_steps = 800
classifier = GeneralizedFuzzyEvolveClassifier(
                evolve_type = evolve_type,
                weighted_loss=[1.0,1.1],
                n_visits = n_bucket,
                report_freq=50,
                patience_step=500,
                max_steps=max_steps,
                learning_rate=0.1,
                batch_size = 64,
                split_method='sample_wise',
                category_info=category_info,
                static_category_info=static_category_info,
                random_state=random_state,
                verbose=2,
                min_epsilon = 0.9,
                sparse_regu=1e-3,
                corr_regu=1e-4,
    
            )
classifier.fit(X_train_variant, y_train,X_train_invariant,
          X_val_variant,X_val_invariant, y_val,)
# pickle.dump(classifier, open(os.path.join(exp_save_path, f'model.mdl'), 'wb'))
train_metrics,threshold = utils.cal_metrics_f1_thresholding(classifier,X_train_variant,X_train_invariant,y_train)
print('auc aucpr F1score')
print('train')
print(train_metrics)
val_metrics = utils.cal_metrics_f1_thresholding(classifier,X_val_variant,X_val_invariant,y_val,0.5)
print('val')
print(val_metrics)
test_metrics = utils.cal_metrics_f1_thresholding(classifier,X_test_variant,X_test_invariant,y_test,0.9)
print('test')
print(test_metrics)


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
static_category_info = static_category_info.astype(np.int32)

time_varying_features = data[:,0:num_time_varying_features*n_bucket]
time_invariant_features = data[:,-num_time_invariant_features:]

report_freq = 50
patience_step = 500
max_steps = 800

# split into training and testing
ss_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
X, y, X_test,y_test = utils.split_dataset(ss_train_test, data, labels, index=0)
X_test = utils.fill_in_missing_value(X_test,X_test)
X_test_variant,X_test_invariant = utils.handle_data(X_test,num_time_invariant_features,n_bucket)
X = utils.fill_in_missing_value(X,X)
X_variant,X_invariant = utils.handle_data(X,num_time_invariant_features,n_bucket)
class params(object):
    binary_pos_only = True
    n_rules = classifier.n_rules
    n_classes = classifier.n_classes
    epsilon = 0.9
    category_info = np.concatenate([category_info,static_category_info]).astype(np.int32)
    feature_names = np.concatenate([feature_names,static_feature_names])
def f(value, epsilon):
    # Re-formulated membership function in this proposed algorithm
    # find indices where value >0 and <0
    i1 = value>=0
    i2 = value<0
    out = np.zeros(value.shape)
    out[i1] = value[i1] + epsilon*np.log(np.exp(-value[i1]/epsilon)+1)
    out[i2] = epsilon*np.log(1+np.exp(value[i2]/epsilon))
    return out

def draw_membership_function(a1, a2, b1, b2, output_path = './images', variable_name='x', n_points=100, epsilon=0,no_zero = True):
    # Build membership function
    start = a1-2*(b1-a2)
    end = b2+2*(b1-a2)
    step = (end-start)/n_points
    x = start + np.array(range(n_points))*step
    if no_zero:
        x = x[x>=0]
    else:
        x = x

    y_low = f((a2-x)/(a2-a1), epsilon) - f((a1-x)/(a2-a1), epsilon)
    y_medium = f((x-a1)/(a2-a1), epsilon) - f((x-a2)/(a2-a1), epsilon) + f((b2-x)/(b2-b1), epsilon) - f((b1-x)/(b2-b1), epsilon) - 1
    y_high = f((x-b1)/(b2-b1), epsilon) - f((x-b2)/(b2-b1), epsilon)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y_low, label='Low')
    plt.plot(x, y_medium, label='Medium')
    plt.plot(x, y_high, label='High')

    plt.legend(fontsize=16)
    plt.ylabel('Membership Value', fontsize=18)
    plt.xlabel(variable_name, fontsize=18)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.savefig(os.path.join(output_path,f'membrship_{variable_name}_{epsilon:.2f}.png'), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    return x,y_low,y_medium,y_high

scaler = classifier.scaler
static_scaler = classifier.static_scaler
net = classifier.estimator


def reverse_log_transform(x):
    return np.exp(x)-1

def extract_encoding_intervals(net, scaler):
    """ Extract the threshold for variable encoding.

    Parameters
    ----------
    net : Net object. Trained network.
    train_features : np.ndarray. Training features used to train the model. It is used to
        calculate the mean and variance we used for data pre-processing.

    Returns
    -------
    encoding_values: np.ndarray. Each row givens the 'fuzzy' threshold for low level, medium level, and high level.

    """
    a2_list = net.weight[0,:] - net.weight[1,:].pow(2)
    b1_list = net.weight[0,:] + net.weight[1,:].pow(2)
    a1_list = a2_list - 0.9 - net.weight[2,:].pow(2)
    b2_list = b1_list + 0.9 + net.weight[3,:].pow(2)

    low_value = scaler.inverse_transform(a1_list.detach().numpy().reshape([1, -1]))
    medium_left = scaler.inverse_transform(a2_list.detach().numpy().reshape([1, -1]))
    medium_right = scaler.inverse_transform(b1_list.detach().numpy().reshape([1, -1]))
    high_value = scaler.inverse_transform(b2_list.detach().numpy().reshape([1, -1]))

    medium_value = (medium_left + medium_right)*0.5
    encoding_values_for_visual = np.concatenate([low_value, medium_value, high_value], axis=0)
    encoding_values = np.concatenate([low_value, medium_left, medium_right, high_value], axis=0)

    return encoding_values_for_visual, encoding_values

encoding_values, extract_encoding_details = extract_encoding_intervals(net.layer_encode, scaler[3])
static_encoding_values, extract_static_encoding_details = extract_encoding_intervals(net.static_layer_encode,static_scaler)
encoding_values = np.concatenate([encoding_values,static_encoding_values],axis = -1)
extract_encoding_details = np.concatenate([extract_encoding_details,extract_static_encoding_details],axis = -1)

encoding_column_continous = np.expand_dims(encoding_values.flatten('F'), axis=1)
encoding_column_categorical = []
category_levels = params.category_info[params.category_info>0]
for n_levels in category_levels:
    encoding_column_categorical += [i for i in range(n_levels)]
encoding_column_categorical = np.expand_dims(np.array(encoding_column_categorical), axis=-1)
encoding_column = np.concatenate([encoding_column_continous,
                                  encoding_column_categorical], axis=0)
continous_variable_name = [params.feature_names[index] \
                                   for index in range(len(params.category_info)) \
                                    if params.category_info[index]==0]

encoding = pd.DataFrame(np.transpose(extract_encoding_details), index=continous_variable_name,
                                columns=['low', 'medium_left', 'medium_right', 'high'])

output_path = './images'
for v_i, var_name in enumerate(encoding.index):
        draw_membership_function(a1 = encoding.iloc[v_i, 0],
                         a2 = encoding.iloc[v_i, 1],
                         b1 = encoding.iloc[v_i, 2],
                         b2 = encoding.iloc[v_i, 3],
                         output_path = output_path,
                         variable_name=encoding.index[v_i], n_points=1000, epsilon=0.1,no_zero = True)




attention_continuous = net.attention_masks_continuous.detach().numpy()
attention_categorical = net.attention_masks_categorical.detach().numpy()
static_attention_continuous = net.static_attention_masks_continuous.detach().numpy()
static_attention_categorical = net.static_attention_masks_categorical.detach().numpy()
connection_mask = net.connection.detach().numpy()

# Convert the weights to masks by applying the tanh function
if params.binary_pos_only:
    static_attention_mask_continous = np.reshape((np.tanh(static_attention_continuous)+1)/2, [-1, params.n_rules])
    static_attention_mask_categorical = np.reshape((np.tanh(static_attention_categorical)+1)/2, [-1, params.n_rules])
    attention_mask_continous = np.reshape((np.tanh(attention_continuous)+1)/2, [-1, params.n_rules])
    attention_mask_categorical = np.reshape((np.tanh(attention_categorical)+1)/2, [-1, params.n_rules])
else:
    static_attention_mask_continous = np.reshape((np.tanh(static_attention_continuous)+1)/2, [-1, params.n_rules*params.n_classes])
    static_attention_mask_categorical = np.reshape((np.tanh(static_attention_categorical)+1)/2, [-1, params.n_rules*params.n_classes])
    attention_mask_continous = np.reshape((np.tanh(attention_continuous)+1)/2, [-1, params.n_rules*params.n_classes])
    attention_mask_categorical = np.reshape((np.tanh(attention_categorical)+1)/2, [-1, params.n_rules*params.n_classes])


attention_mask_continous = np.concatenate([attention_mask_continous,static_attention_mask_continous],axis = 0)
attention_mask_categorical = np.concatenate([attention_mask_categorical,static_attention_mask_categorical],axis = 0)


# In the connection matrix, each entry indicates the contribution of one variable.
# As a result, we need to expand the contribution of one variable to match the number
# of concepts from this variable.
connection_mat_continous = []
connection_mat_categorical = []

# all_category_info = np.concatenate([params.category_info,params.static_category_info]).astype(np.int32)

num_continous_x = np.sum(params.category_info==0)
temp_num_categorical_x = 0
xx = 0
for i in range(len(params.category_info)):
    if params.category_info[i] == 0:
        xx +=1
        connection_mat_continous += [connection_mask[i-temp_num_categorical_x,:]]*3
    else:
        connection_mat_categorical += [connection_mask[num_continous_x+temp_num_categorical_x,:]]*params.category_info[i]
        temp_num_categorical_x += 1

connection_mat_continous = np.stack(connection_mat_continous, axis=0)
relation_mat_continous = attention_mask_continous*connection_mat_continous

if len(connection_mat_categorical)>0:
    connection_mat_categorical = np.stack(connection_mat_categorical, axis=0)
    relation_mat_categorical = attention_mask_categorical*connection_mat_categorical

    attention_mask = np.concatenate([attention_mask_continous, attention_mask_categorical], axis=0)
    relation_mat = np.concatenate([relation_mat_continous, relation_mat_categorical], axis=0)
else:
     attention_mask, connection_mask,relation_mat = attention_mask_continous, connection_mask, relation_mat_continous
     
row_names_continous = []
row_names_categorical = []
for i in range(len(params.category_info)):
    if params.feature_names is None:
        if params.category_info[i] == 0:
            # row_names_continous += [f'x{i}_low', f'x{i}_medium', f'x{i}_high']
            row_names_continous += [f'low x{i}', f'medium x{i}', f'high x{i}']
        else:
            row_names_categorical += [f'x{i}_level{j}' for j in range(params.category_info[i])]
    else:
        feature_name = params.feature_names[i]
        if params.category_info[i] == 0:
            # row_names_continous += [f'{feature_name}_low', f'{feature_name}_medium', f'{feature_name}_high']
            row_names_continous += [f'Low {feature_name}', f'Medium {feature_name}', f'High {feature_name}']

        elif params.category_info[i] > 2:
            row_names_categorical += [f'{feature_name}_level{j}' for j in range(params.category_info[i])]


        elif params.category_info[i] == 2:
            # row_names_categorical += [f'{feature_name}_level{j}' for j in range(params.category_info[i])]
            for j in range(params.category_info[i]):
                if j == 0:
                    row_names_categorical.append(f'No {feature_name}')
                elif j == 1:
                    row_names_categorical.append(f'Exist {feature_name}')
row_names = row_names_continous + row_names_categorical

if params.binary_pos_only:
    out_layer = net.layer_inference.weight.detach().numpy()**2
else:
    out_layer = net.layer_inference.weight.detach().numpy()**2
    out_layer[:params.n_rules] *= -1


weighted_out_layer = out_layer/max(np.abs(out_layer))
out_row = np.insert(weighted_out_layer, 0, np.nan)

rules = np.concatenate([encoding_column, relation_mat], axis=-1)
rules = np.concatenate([rules, np.expand_dims(out_row, axis=0)], axis=0)
row_names.append('direction')

def remove_redundant_rules(mat, params, information_threshold=0.01, similarity_threshold=0.9):
    """ Filter redundant rules.

    If the information of one rule is very small (sum of the concepts' contribution in this rule),
    the rule will be removed.

    If we find a group of rules with similar pattern, the rule with largest weight in the inference layer
    will be selected.
    """
    relation_mat = mat[:-1, 1:]
    rule_weight = mat[-1, 1:]

    # Remove rules with low weight to the output layer
    relation_list = []
    rule_weight_list = []
    for i in range(params.n_rules):
        if (rule_weight[i] >= information_threshold) or (np.sum(relation_mat[:,i]) <= information_threshold):
            relation_list.append(relation_mat[:, i])
            rule_weight_list.append(rule_weight[i])
    relation_mat = np.stack(relation_list, axis=-1)
    weights = np.array(rule_weight_list)

    df = pd.DataFrame(relation_mat)
    corr_mat = df.corr()

    rule_index_list = list(range(relation_mat.shape[1]))
    keep_rule_index_list = list(range(relation_mat.shape[1]))

    rule_groups = []
    while len(rule_index_list)>0:
        index = rule_index_list.pop(0)
        # If the rule value is not a constant
        if np.max(relation_mat[:, index])>information_threshold:
            corr = corr_mat.iloc[index, :]
            merge_list = [index]
            for j in range(index+1, corr.shape[0]):
                if corr[j] > similarity_threshold and j in keep_rule_index_list:
                    keep_rule_index_list.remove(j)
                    rule_index_list.remove(j)
                    merge_list.append(j)
            rule_groups.append(merge_list)

    rule_index_list = []
    for merge_list in rule_groups:
        weight_list = np.take(weights, merge_list)
        # Test whether have the same direction. If not, rules in this group should be removed.
        if np.all(weight_list == np.abs(weight_list)) or np.all(weight_list == -np.abs(weight_list)):
            sel_index = merge_list[np.argmax(np.abs(weight_list))]
            rule_index_list.append(sel_index)

    relation_mat = np.concatenate([relation_mat, np.expand_dims(weights, axis=0)], axis=0)
    filtered_rules = np.take(relation_mat, rule_index_list, axis=1)
    encoding = mat[:, 0:1]
    mat = np.concatenate([encoding, filtered_rules], axis=1)
    return mat, rule_index_list

merged_rules, keep_index = remove_redundant_rules(rules, params,
                                                      information_threshold=0.1,
                                                      similarity_threshold=0.8)

if params.binary_pos_only:
    all_column_names = ['encoding']
    for i in range(params.n_rules):
        all_column_names.append(f'Rule_{i}')
else:
    all_column_names = ['encoding']
    for i in range(params.n_classes*params.n_rules):
        all_column_names.append(f'Rule_{i}')

column_names = ['encoding']
for i, index in enumerate(keep_index):
    column_names.append(f'Rule_{i+1}')
    
def filter_irrelevant_variables(mat, row_names, columns, category_info, filter_threshold=0.1):
    """ Filter the variables with little conribution to the classification.

    The contribution of each variables (including all concepts in all rules) will be summed up.
    Then the variables with contribution smaller than the threshold will be filtered out.

    Parameters
    ----------
    mat : np.ndarray. Extracted rules.
    row_names : A list. Name of variables concepts for the mat.
    columns: A list. Column names.
    params : Params object.
    filter_threshold : A float. The threshold used to filter the irrelevant varaible.
    Returns
    -------
    filtered_rule_mat: np.ndarray. Extracted rules after irrelevant variables are removed.
    filtered_row_names : A list of strings. Name of variables concepts after irrelevant variables are removed.
    """

    filtered_row_names = []
    filtered_mat = []
    n_continous_variables = np.sum(category_info==0)
    n_variables = len(category_info)
    category_levels = category_info[category_info>0]

    for i in range(n_variables):
        if i<n_continous_variables:
            submat = mat[i*3:(i+1)*3, 1:]
        else:
            prev = n_continous_variables*3+np.sum(category_levels[:i-n_continous_variables])
            submat = mat[prev:prev+category_levels[i-n_continous_variables], 1:]

        min_value = np.min(submat, axis=0)
        submat -= min_value

        if np.max(submat) > filter_threshold:
            if i < n_continous_variables:
                filtered_mat.append(np.concatenate([mat[i*3:(i+1)*3, 0:1], submat], axis=1))
                filtered_row_names += row_names[i*3:(i+1)*3]
            else:
                filtered_mat.append(np.concatenate([mat[prev:prev+category_levels[i-n_continous_variables], 0:1], submat], axis=1))
                filtered_row_names += row_names[prev:prev+category_levels[i-n_continous_variables]]

        else:
            if i<n_continous_variables:
                filtered_variable_name = row_names[i*3].split(' ')[1]
            else:
                prev = n_continous_variables*3+np.sum(category_levels[:i-n_continous_variables])
                filtered_variable_name = row_names[prev].split(' ')[1]
            print(f'{filtered_variable_name} does not contribute to rules.')

    if len(filtered_mat)==0:
        print('All variables are filtered.')
        rules =  pd.DataFrame(mat, columns=columns, index=row_names)
        return rules
    else:
        filtered_rule_mat = np.concatenate(filtered_mat, axis=0)
        filtered_rule_mat = np.concatenate([filtered_rule_mat, mat[-1:,:]], axis=0)
        filtered_row_names.append('directions')
        rules =  pd.DataFrame(filtered_rule_mat, columns=columns, index=filtered_row_names)
        return rules
    
rules = filter_irrelevant_variables(merged_rules, row_names, column_names, params.category_info,
                                                        filter_threshold=0.01)
import seaborn as sns
import matplotlib.pyplot as plt
table = rules
rules = np.round(table.iloc[:-1,1:],3)
directions = table.iloc[-1, 1:]
directions = np.round(directions,2)


columns = rules.columns
new_columns = []
for i in range(columns.shape[0]):
    new_columns.append('\n'.join([columns[i].replace('_', ' '), str(directions[i])]))
    # new_columns.extend([columns[i].replace('_', ' ')])
# print(new_columns)
rules.columns = new_columns
n_height = 1
n_weight = 1
plt.figure(figsize=(n_weight*rules.shape[1], rules.shape[0]//3*n_height))
sns.heatmap(rules, cmap="rocket_r")
import datetime

# Get the current date and time
current_time = datetime.datetime.now()

# Generate a timestamp string in the format: YYYY-MM-DD-HH-MM-SS
timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")

# Create the image file name with the timestamp

image_file_name = os.path.join(output_path,f"image_{timestamp}.png")
plt.savefig(image_file_name , dpi=300, bbox_inches="tight")