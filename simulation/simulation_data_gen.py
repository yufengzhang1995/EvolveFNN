from __future__ import absolute_import, division, print_function

import numpy as np


def generate_simluated_time_series_data(n_samples,n_timestamp,mislabel = None,random_state=1234):
  np.random.seed(random_state)
  x0 = 2**0.5*np.random.randn(n_samples, 1)
  x1 = 3**0.5*np.random.randn(n_samples, 1) + 5
  x2 = 5**0.5*np.random.randn(n_samples, 1) - 1
  x3 = 2**0.5*np.random.randn(n_samples, 1) + 1
  x4 = np.random.randn(n_samples, 1) - 2
  x_noise = np.random.randn(n_samples, 2)
  dynamic_features = np.concatenate([x0, x1, x2, x3, x4, x_noise], axis=-1)

  x1_static = np.expand_dims(np.random.randint(2, size=n_samples), axis=-1)
  x2_static = np.expand_dims(np.random.randint(10, size=n_samples), axis=-1)
  static_features = np.concatenate([x1_static,x2_static], axis=-1)

  timestamps = np.arange(n_timestamp)
  time_series_data = np.zeros((n_samples,n_timestamp,dynamic_features.shape[-1]))
  for i in range(n_samples):
      input_i = dynamic_features[i,:]
      input_i = input_i[:, np.newaxis] + np.cos((2 * input_i[:, np.newaxis] + timestamps[np.newaxis, :]))
      time_series_data[i] = input_i.T

  x0_last = time_series_data[:,-1,0].reshape((-1,1))
  x1_last = time_series_data[:,-1,1].reshape((-1,1))
  x2_last = time_series_data[:,-1,2].reshape((-1,1))
  x3_last = time_series_data[:,-1,3].reshape((-1,1))
  x4_last = time_series_data[:,-1,4].reshape((-1,1))

  rules = [np.logical_and.reduce([x1_last<3.8, x2_last>-2,x1_static == 1], axis=0),
              np.logical_and.reduce([x1_last>6.3, x2_last>-2,x2_static > 6], axis=0),
              np.logical_and.reduce([x0_last<1, x3_last>2], axis=0),
              np.logical_and.reduce([x2_last>0, x4_last>-1,x1_static == 1], axis=0),
              np.logical_and.reduce([x0_last<1, x4_last>-1.5,x2_static > 6], axis=0)]
    
  for i in range(len(rules)):
      print('Rule {}: {:.2f}%'.format(i, np.sum(rules[i])/n_samples*100))

  labels = np.logical_or.reduce(rules, axis=0)[:,0]
  if mislabel is not None:
      one_array = labels[labels==1]
      mutated_one_array = np.where(np.random.random(one_array.shape) < mislabel, False, one_array)
      labels[labels==1] = mutated_one_array


  categorical_info = np.zeros([dynamic_features.shape[1]]) 
  print('Positive samples: {:.2f}%'.format(np.sum(labels)/dynamic_features.shape[0]*100))
  return time_series_data,static_features, labels

