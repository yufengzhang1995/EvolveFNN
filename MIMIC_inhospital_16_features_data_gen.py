import os
import numpy as np
import random
import pandas as pd

import platform
import pickle
import json
from MIMIC_utils import *

import torch
from torch.utils.data import Dataset

def convert_Capillary(df):
    df['Cat_Capillary_refill_rate'] = df.apply(lambda row: 1 if row['Capillary refill rate->0.0'] == 1 else 0, axis=1)
    df.drop(columns=['Capillary refill rate->0.0', 'Capillary refill rate->1.0'], inplace=True)

def convert_eye_opening(df):
    def f(row):
        if row['Glascow coma scale eye opening->None'] == 1:
            return 0
        elif row['Glascow coma scale eye opening->1 No Response'] == 1:
            return 1
        elif row['Glascow coma scale eye opening->To Pain'] == 1:
            return 2
        elif row['Glascow coma scale eye opening->2 To pain'] == 1:
            return 2
        elif row['Glascow coma scale eye opening->To Speech'] == 1:
            return 3
        elif row['Glascow coma scale eye opening->3 To speech'] == 1:
            return 3
        elif row['Glascow coma scale eye opening->Spontaneously'] == 1:
            return 4
        elif row['Glascow coma scale eye opening->4 Spontaneously'] == 1:
            return 4
    df['Glascow_coma_scale_eye_opening'] = df.apply(lambda row: f(row), axis=1)
    df.drop(columns=['Glascow coma scale eye opening->To Pain',
                 'Glascow coma scale eye opening->3 To speech',
                 'Glascow coma scale eye opening->1 No Response',
                 'Glascow coma scale eye opening->4 Spontaneously',
                 'Glascow coma scale eye opening->None',
                 'Glascow coma scale eye opening->To Speech',
                 'Glascow coma scale eye opening->Spontaneously',
                 'Glascow coma scale eye opening->2 To pain',], inplace=True)

def convert_motor_response(df):
    def f(row):
        if row['Glascow coma scale motor response->1 No Response'] == 1:
            return 0
        elif row['Glascow coma scale motor response->No response'] == 1:
            return 0
        elif row['Glascow coma scale motor response->2 Abnorm extensn'] == 1:
            return 1
        elif row['Glascow coma scale motor response->Abnormal extension'] == 1:
            return 1
        elif row['Glascow coma scale motor response->3 Abnorm flexion'] == 1:
            return 2
        elif row['Glascow coma scale motor response->Abnormal Flexion'] == 1:
            return 2
        elif row['Glascow coma scale motor response->4 Flex-withdraws'] == 1:
            return 3
        elif row['Glascow coma scale motor response->Flex-withdraws'] == 1:
            return 3
        elif row['Glascow coma scale motor response->5 Localizes Pain'] == 1:
            return 4
        elif row['Glascow coma scale motor response->Localizes Pain'] == 1:
            return 4
        elif row['Glascow coma scale motor response->6 Obeys Commands'] == 1:
            return 5
        elif row['Glascow coma scale motor response->Obeys Commands'] == 1:
            return 5
    df['Glascow_coma_scale_motor_response'] = df.apply(lambda row: f(row), axis=1)
    df.drop(columns=['Glascow coma scale motor response->1 No Response',
             'Glascow coma scale motor response->3 Abnorm flexion',
             'Glascow coma scale motor response->Abnormal extension',
             'Glascow coma scale motor response->No response',
             'Glascow coma scale motor response->4 Flex-withdraws',
             'Glascow coma scale motor response->Localizes Pain',
             'Glascow coma scale motor response->Flex-withdraws',
             'Glascow coma scale motor response->Obeys Commands',
             'Glascow coma scale motor response->Abnormal Flexion',
             'Glascow coma scale motor response->6 Obeys Commands',
             'Glascow coma scale motor response->5 Localizes Pain',
             'Glascow coma scale motor response->2 Abnorm extensn',], inplace=True)
def convert_scale_total(df):
    def f(row):
        if row['Glascow coma scale total->11'] == 1:
            return 11
        elif row['Glascow coma scale total->10'] == 1:
            return 10
        elif row['Glascow coma scale total->13'] == 1:
            return 13
        elif row['Glascow coma scale total->12'] == 1:
            return 12
        elif row['Glascow coma scale total->15'] == 1:
            return 15
        elif row['Glascow coma scale total->14'] == 1:
            return 14
        elif row['Glascow coma scale total->3'] == 1:
            return 3
        elif row['Glascow coma scale total->5'] == 1:
            return 5
        elif row['Glascow coma scale total->4'] == 1:
            return 4
        elif row['Glascow coma scale total->7'] == 1:
            return 7
        elif row['Glascow coma scale total->6'] == 1:
            return 6
        elif row['Glascow coma scale total->9'] == 1:
            return 9
        elif row['Glascow coma scale total->8'] == 1:
            return 8
    df['Glascow_coma_scale_total'] = df.apply(lambda row: f(row), axis=1)
    df.drop(columns=['Glascow coma scale total->11',
                 'Glascow coma scale total->10',
                 'Glascow coma scale total->13',
                 'Glascow coma scale total->12',
                 'Glascow coma scale total->15',
                 'Glascow coma scale total->14',
                 'Glascow coma scale total->3',
                 'Glascow coma scale total->5',
                 'Glascow coma scale total->4',
                 'Glascow coma scale total->7',
                 'Glascow coma scale total->6',
                 'Glascow coma scale total->9',
                 'Glascow coma scale total->8',], inplace=True)

def convert_verbal_response(df):
    def f(row):
        if row['Glascow coma scale verbal response->1 No Response'] == 1:
            return 0
        elif row['Glascow coma scale verbal response->No Response'] == 1:
            return 0
        elif row['Glascow coma scale verbal response->Confused'] == 1:
            return 3
        elif row['Glascow coma scale verbal response->4 Confused'] == 1:
            return 3
        elif row['Glascow coma scale verbal response->Inappropriate Words'] == 1:
            return 2
        elif row['Glascow coma scale verbal response->3 Inapprop words'] == 1:
            return 2
        elif row['Glascow coma scale verbal response->Oriented'] == 1:
            return 4
        elif row['Glascow coma scale verbal response->5 Oriented'] == 1:
            return 4
        elif row['Glascow coma scale verbal response->No Response-ETT'] == 1:
            return 0
        elif row['Glascow coma scale verbal response->Incomprehensible sounds'] == 1:
            return 1
        elif row['Glascow coma scale verbal response->2 Incomp sounds'] == 1:
            return 1
        elif row['Glascow coma scale verbal response->1.0 ET/Trach'] == 1:
            return 0
    df['Glascow_coma_verbal_response'] = df.apply(lambda row: f(row), axis=1)
    df.drop(columns=['Glascow coma scale verbal response->1 No Response',
                 'Glascow coma scale verbal response->No Response',
                 'Glascow coma scale verbal response->Confused',
                 'Glascow coma scale verbal response->Inappropriate Words',
                 'Glascow coma scale verbal response->Oriented',
                 'Glascow coma scale verbal response->No Response-ETT',
                 'Glascow coma scale verbal response->5 Oriented',
                 'Glascow coma scale verbal response->Incomprehensible sounds',
                 'Glascow coma scale verbal response->1.0 ET/Trach',
                 'Glascow coma scale verbal response->4 Confused',
                 'Glascow coma scale verbal response->2 Incomp sounds',
                 'Glascow coma scale verbal response->3 Inapprop words'], inplace=True)


def drop_all_mask(df):
    df.drop(columns=['mask->Capillary refill rate',
                 'mask->Diastolic blood pressure',
                 'mask->Fraction inspired oxygen',
                 'mask->Glascow coma scale eye opening',
                 'mask->Glascow coma scale motor response',
                 'mask->Glascow coma scale total',
                 'mask->Glascow coma scale verbal response',
                 'mask->Glucose',
                 'mask->Heart Rate',
                 'mask->Height',
                 'mask->Mean blood pressure',
                 'mask->Oxygen saturation',
                 'mask->Respiratory rate',
                 'mask->Systolic blood pressure',
                 'mask->Temperature',
                 'mask->Weight',
                 'mask->pH'], inplace=True)

def transform_data(array_dat,discretizer_header):
    temp_dat = pd.DataFrame(array_dat,columns = discretizer_header)
    convert_Capillary(temp_dat)
    convert_eye_opening(temp_dat)
    convert_motor_response(temp_dat)
    convert_scale_total(temp_dat)
    convert_verbal_response(temp_dat)
    drop_all_mask(temp_dat)
    temp_array = np.array(temp_dat)
    return temp_array

def get_transformed_data_name(array_dat,discretizer_header):
    temp_dat = pd.DataFrame(array_dat,columns = discretizer_header)
    convert_Capillary(temp_dat)
    convert_eye_opening(temp_dat)
    convert_motor_response(temp_dat)
    convert_scale_total(temp_dat)
    convert_verbal_response(temp_dat)
    drop_all_mask(temp_dat)
    return temp_dat.columns

class InHospitalMortalityReader(Reader):
    def __init__(self, dataset_dir, listfile=None, period_length=48.0):
        """ Reader for in-hospital moratality prediction task.

        :param dataset_dir:   Directory where timeseries files are stored.
        :param listfile:      Path to a listfile. If this parameter is left `None` then
                              `dataset_dir/listfile.csv` will be used.
        :param period_length: Length of the period (in hours) from which the prediction is done.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, int(y)) for (x, y) in self._data]
        self._period_length = period_length

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : int (0 or 1)
                In-hospital mortality.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._period_length
        y = self._data[index][1]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}
        
class InHospitalMortalityDataset(Dataset):
    def __init__(self, dataset_dir, listfile=None):
        """
        PyTorch dataset for phenotype classification task.

        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile: Path to a listfile. If this parameter is left `None` then
                         `dataset_dir/listfile.csv` will be used.
        """
        self.reader = InHospitalMortalityReader(dataset_dir, listfile)

    def __len__(self):
        return self.reader.get_number_of_examples()

    def __getitem__(self, index):
        example = self.reader.read_example(index)
        X = example["X"]
        t = example["t"]
        y = example["y"]
        header = example["header"]
        name = example["name"]
        return X, t, y, header,name

data_root = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Data/Processed/Yufeng/MIMIC_benchmark/in-hospital-mortality/'
timestep = 6
for n in ['train','test']:
    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_root, n),
                             listfile=os.path.join(data_root, n,'listfile.csv'))
    discretizer = Discretizer(timestep=float(timestep), # timestep = 1.0
                            store_masks=True,
                            impute_strategy='previous',
                            start_time='zero')
    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
    # normalizer = Normalizer(fields=cont_channels)
    # normalizer_state = 'ihm_ts{}.input_str-{}.start_time-zero.normalizer'.format(1.0, 'previous')
    # normalizer_state = os.path.join('./resources', normalizer_state)
    # normalizer.load_params(normalizer_state)



    N = train_reader.get_number_of_examples()
    # N = 1200
    print(f'for {n}, the sample size is {N}')
    ret = read_chunk(train_reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    feature_names= get_transformed_data_name(data[0],discretizer_header)
    feature_names = list(feature_names)
    data = [transform_data(array_dat,discretizer_header) for array_dat in  data]

    statci_index = [feature_names.index('Height')]
    dynamic_data = [np.delete(mat, statci_index, axis=1) for mat in data]
    fixed_data = [np.take(mat, statci_index, axis=1) for mat in data]

    dynamic_features_names = [e for e in feature_names if e not in ('Height')]
    static_feature_names = ['Height']

    category_info = [0,0,0,0,0,0,0,0,0,0,0,2,5,6,0,5]
    static_category_info = [0]
    fixed_data = np.array(fixed_data)[:,1,:]

    dataset = {
                    'full_data': None,
                    'variables': np.array(dynamic_data),
                    'static_variables':np.array(fixed_data),
                    'response': labels,
                    'feature_names': dynamic_features_names,
                    'static_feature_names':static_feature_names,
                    'num_classes': 2,
                    'category_info': category_info,
                    'static_category_info': static_category_info,
                    'split_method': 'sample_wise',
                    }
    dataset_root = './MIMIC_inhospital/'
    pickle.dump(dataset, open(os.path.join(dataset_root,f"InHospitalMortality_{n}_sample_{N}_timestep_{timestep}_reduced_features.p"), "wb"))

