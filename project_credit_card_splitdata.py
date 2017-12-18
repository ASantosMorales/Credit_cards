#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 12:27:19 2017

@author: a_santos

Since the data were modified the next step is to split them in training and
testing sets. The data are unvalanced because there are a lot of zero-targets
and only about 20% of one-targets. I have to be careful at the time of spliting data.
This script tries to overcome these problem. 

"""

import numpy as np

from sklearn.cross_validation import train_test_split

#%% Get modified data
path = '/home/a_santos/Documents/TEC de Monterrey/Semestre_3/Receonocimiento de patrones/Proyecto_Final/Data_preprocessing/data_modified.npy' 
data = np.load(path)
#%% Arrangement data (unvalanced data)
data_ones = []
data_zeros = []
for i in range(len(data)):
    if data[i, 23] == 1:
        data_ones.append(data[i, :])
    else:
        data_zeros.append(data[i, :])

# Spliting data     
data_ones_split_train_1, data_ones_split_test_1 = train_test_split(data_ones, test_size=0.4)
data_ones_split_train_2, data_ones_split_test_2 = train_test_split(data_ones, test_size=0.3)
data_ones_split_train_3, data_ones_split_test_3 = train_test_split(data_ones, test_size=0.2)
data_ones_split_train_4, data_ones_split_test_4 = train_test_split(data_ones, test_size=0.1)

data_zeros_split_train_1, data_zeros_split_test_1 = train_test_split(data_zeros, test_size=0.4)
data_zeros_split_train_2, data_zeros_split_test_2 = train_test_split(data_zeros, test_size=0.3)
data_zeros_split_train_3, data_zeros_split_test_3 = train_test_split(data_zeros, test_size=0.2)
data_zeros_split_train_4, data_zeros_split_test_4 = train_test_split(data_zeros, test_size=0.1)

# Concatenate and randomization of data
data_train_1 = np.vstack((data_ones_split_train_1, data_zeros_split_train_1))
np.random.shuffle(data_train_1)
np.random.shuffle(data_train_1)
data_train_2 = np.vstack((data_ones_split_train_2, data_zeros_split_train_2))
np.random.shuffle(data_train_2)
np.random.shuffle(data_train_2)
data_train_3 = np.vstack((data_ones_split_train_3, data_zeros_split_train_3))
np.random.shuffle(data_train_3)
np.random.shuffle(data_train_3)
data_train_4 = np.vstack((data_ones_split_train_4, data_zeros_split_train_4))
np.random.shuffle(data_train_4)
np.random.shuffle(data_train_4)

data_test_1 = np.vstack((data_ones_split_test_1, data_zeros_split_test_1))
np.random.shuffle(data_test_1)
np.random.shuffle(data_test_1)
data_test_2 = np.vstack((data_ones_split_test_2, data_zeros_split_test_2))
np.random.shuffle(data_test_2)
np.random.shuffle(data_test_2)
data_test_3 = np.vstack((data_ones_split_test_3, data_zeros_split_test_3))
np.random.shuffle(data_test_3)
np.random.shuffle(data_test_3)
data_test_4 = np.vstack((data_ones_split_test_4, data_zeros_split_test_4))
np.random.shuffle(data_test_4)
np.random.shuffle(data_test_4)

#%% Target data distribution comprobation
print(sum(data_train_1[:, 23])/len(data_train_1))
print(sum(data_train_2[:, 23])/len(data_train_2))
print(sum(data_train_3[:, 23])/len(data_train_3))
print(sum(data_train_4[:, 23])/len(data_train_4))

print(sum(data_test_1[:, 23])/len(data_test_1))
print(sum(data_test_2[:, 23])/len(data_test_2))
print(sum(data_test_3[:, 23])/len(data_test_3))
print(sum(data_test_4[:, 23])/len(data_test_4))

#%% Save data
np.save('data_train_1.npy', data_train_1)
np.save('data_train_2.npy', data_train_1)
np.save('data_train_3.npy', data_train_2)
np.save('data_train_4.npy', data_train_3)

np.save('data_test_1.npy', data_test_1)
np.save('data_test_2.npy', data_test_2)
np.save('data_test_3.npy', data_test_3)
np.save('data_test_4.npy', data_test_4)