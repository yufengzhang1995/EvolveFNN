import os
import numpy as np
import random


import platform
import pickle
import json
from MIMIC_utils import *

import torch
from torch.utils.data import Dataset



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

    statci_index = [discretizer_header.index('Height'), discretizer_header.index('mask->Height'),]
    dynamic_data = [np.delete(mat, statci_index, axis=1) for mat in data]
    fixed_data = [np.take(mat, statci_index, axis=1) for mat in data]


    dynamic_features = [e for e in discretizer_header if e not in ('Height','mask->Height')]
    dynamic_cont = [i for (i, x) in enumerate(dynamic_features) if x.find("->") == -1]
    static_feature_names = ['Height','mask->Height']



    # data = [normalizer.transform(X) for X in data]
    category_info = np.zeros(len(dynamic_features))+2
    category_info[dynamic_cont] = 0
    static_category_info = [0,2]
    
    
    fixed_data = np.array(fixed_data)[:,1,:]
    fixed_data[:,-1] = 0
    dataset = {
                    'full_data': None,
                    'variables': np.array(dynamic_data),
                    'static_variables':np.array(fixed_data),
                    'response': labels,
                    'feature_names': discretizer_header,
                    'static_feature_names':static_feature_names,
                    'num_classes': 2,
                    'category_info': category_info,
                    'static_category_info': static_category_info,
                    'split_method': 'sample_wise',
                    }
    dataset_root = './MIMIC_inhospital/'
    pickle.dump(dataset, open(os.path.join(dataset_root,f"InHospitalMortality_{n}_sample_{N}_timestep_{timestep}.p"), "wb"))