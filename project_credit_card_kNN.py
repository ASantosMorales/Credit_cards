#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 19:30:55 2017

@author: a_santos

Solving the Default payment of credit cards problem with the kNN algorithm.

* Here is a systematic evaluation to find the best kNN parameters
  to predict the default payment of credit cards.
* This script needs the preprocessed data obtained on the 
  project_credit_card_splitdata.py script.
* The best parameters will be k = 3 and p = 2.

"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

#%% Get the preprocessed data
path_data_train_1 = 'Write the corresponding .npy file path here'
path_data_train_2 = 'Write the corresponding .npy file path here'
path_data_train_3 = 'Write the corresponding .npy file path here'
path_data_train_4 = 'Write the corresponding .npy file path here'

path_data_test_1 = 'Write the corresponding .npy file path here'
path_data_test_2 = 'Write the corresponding .npy file path here'
path_data_test_3 = 'Write the corresponding .npy file path here'
path_data_test_4 = 'Write the corresponding .npy file path here'

data_train_1 = np.load(path_data_train_1)
data_train_2 = np.load(path_data_train_2)
data_train_3 = np.load(path_data_train_3)
data_train_4 = np.load(path_data_train_4)

data_test_1 = np.load(path_data_test_1)
data_test_2 = np.load(path_data_test_2)
data_test_3 = np.load(path_data_test_3)
data_test_4 = np.load(path_data_test_4)

# Assign the characteristics vectors and the target vectors
X_train_1 = data_train_1[:, 0:23]
X_train_2 = data_train_2[:, 0:23]
X_train_3 = data_train_3[:, 0:23]
X_train_4 = data_train_4[:, 0:23]

X_test_1 = data_test_1[:, 0:23]
X_test_2 = data_test_2[:, 0:23]
X_test_3 = data_test_3[:, 0:23]
X_test_4 = data_test_4[:, 0:23]

Y_train_1 = data_train_1[:, 23]
Y_train_2 = data_train_2[:, 23]
Y_train_3 = data_train_3[:, 23]
Y_train_4 = data_train_4[:, 23]

Y_test_1 = data_test_1[:, 23]
Y_test_2 = data_test_2[:, 23]
Y_test_3 = data_test_3[:, 23]
Y_test_4 = data_test_4[:, 23]

# 
del(data_train_1)
del(data_train_2)
del(data_train_3)
del(data_train_4)

del(data_test_1)
del(data_test_2)
del(data_test_3)
del(data_test_4)

del(path_data_train_1)
del(path_data_train_2)
del(path_data_train_3)
del(path_data_train_4)

del(path_data_test_1)
del(path_data_test_2)
del(path_data_test_3)
del(path_data_test_4)

# Varible lists creation
X_train_list = ['X_train_1', 'X_train_2', 'X_train_3', 'X_train_4']
Y_train_list = ['Y_train_1', 'Y_train_2', 'Y_train_3', 'Y_train_4']
mse_train_list = ['mse_train_1', 'mse_train_2', 'mse_train_3', 'mse_train_4']
success_rate_train_list = ['success_rate_train_1', 'success_rate_train_2', 'success_rate_train_3', 'success_rate_train_4']

X_test_list = ['X_test_1', 'X_test_2', 'X_test_3', 'X_test_4']
Y_test_list = ['Y_test_1', 'Y_test_2', 'Y_test_3', 'Y_test_4']

#%% Best parameters determination.

metrics = np.array([1, 2, 5])
neighbors = np.array([3, 5, 7, 9])

for n in range(4):
    print('New running')
    X_train = eval(X_train_list[n])
    Y_train = eval(Y_train_list[n])
    success_rate = np.zeros([len(neighbors), len(metrics)])
    mse = np.zeros([len(neighbors), len(metrics)])
    m = 0
    for k in metrics:
        l = 0
        for j in neighbors:
            kNN = KNeighborsClassifier(n_neighbors=j, p=k)
            kNN.fit(X_train, Y_train)
            Y_train_out = np.zeros([len(Y_train), 1])
            for i in range(len(Y_train)):
                Y_train_out[i] = kNN.predict([X_train[i, :]])
            cnf_matrix_train = confusion_matrix(Y_train, Y_train_out)
            success_rate[l, m] = ((cnf_matrix_train[0, 0] + cnf_matrix_train[1, 1]) / \
                        (cnf_matrix_train[0, 0] + cnf_matrix_train[0, 1] + \
                         cnf_matrix_train[1, 0] + cnf_matrix_train[1, 1]))
            mse[l, m] = (mean_squared_error(Y_train, Y_train_out))
            del(kNN)
            print('metric = ', metrics[m], '  neighbor = ', neighbors[l])
            l = l + 1
        m = m + 1
    globals()[success_rate_train_list[n]] = success_rate
    globals()[mse_train_list[n]] = mse

            
#%% Implementation with k = 3 and p = 2
cnf_matrix_test = np.zeros([2, 2, 4])
success_rate = np.zeros([4, 1])
mse = np.zeros([4, 1])
for i in range(4):
    X_train = eval(X_train_list[i])
    Y_train = eval(Y_train_list[i])
    X_test = eval(X_test_list[i])
    Y_test = eval(Y_test_list[i])
    Y_test_out = np.zeros([len(Y_test), 1])
    kNN = KNeighborsClassifier(n_neighbors=3, p=2)
    kNN.fit(X_train, Y_train)
    for j in range(len(Y_test)):
        Y_test_out[j] = kNN.predict([X_test[j, :]])
    cnf_matrix_test[:, :, i] = confusion_matrix(Y_test, Y_test_out)
    success_rate[i] = ((cnf_matrix_test[0, 0, i] + cnf_matrix_test[1, 1, i]) / \
                (cnf_matrix_test[0, 0, i] + cnf_matrix_test[0, 1, i] + \
                 cnf_matrix_test[1, 0, i] + cnf_matrix_test[1, 1, i]))
    mse[i] = (mean_squared_error(Y_test, Y_test_out))
    del(kNN)
    print('running = ', i)

#%%*****************************************************************************
# PConfusion matrices plotting
class_names = ['YES', 'NO']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#*****************************************************************************
plt.figure()
plt.subplot(221)
plot_confusion_matrix(cnf_matrix_test[:, :, 0].astype(int), classes=class_names,
                      title='Default payments / Test data / run 1')

plt.subplot(222)
plot_confusion_matrix(cnf_matrix_test[:, :, 1].astype(int), classes=class_names,
                      title='Default payments / Test data / run 2')

plt.subplot(223)
plot_confusion_matrix(cnf_matrix_test[:, :, 2].astype(int), classes=class_names,
                      title='Default payments / Test data / run 2')

plt.subplot(224)
plot_confusion_matrix(cnf_matrix_test[:, :, 3].astype(int), classes=class_names,
                      title='Default payments / Test data / run 2')
