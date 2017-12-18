#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:17:40 2017

@author: a_santos

Solving the Default payment of credit cards problem with Support vector
machine algorithm (linear kernel).

* Here is a systematic evaluation to find the best SVM parameter C
  to predict the default payment of credit cards.
* This script needs the preprocessed data obtained on the 
  project_credit_card_splitdata.py script.
* The best parameters will be C = 1.

"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

#%% Get the preprocessed data
path_data_train_1 = '/home/a_santos/Documents/TEC de Monterrey/Semestre_3/Receonocimiento de patrones/Proyecto_Final/Data_preprocessing/data_train_1.npy'
path_data_train_2 = '/home/a_santos/Documents/TEC de Monterrey/Semestre_3/Receonocimiento de patrones/Proyecto_Final/Data_preprocessing/data_train_2.npy'
path_data_train_3 = '/home/a_santos/Documents/TEC de Monterrey/Semestre_3/Receonocimiento de patrones/Proyecto_Final/Data_preprocessing/data_train_3.npy'
path_data_train_4 = '/home/a_santos/Documents/TEC de Monterrey/Semestre_3/Receonocimiento de patrones/Proyecto_Final/Data_preprocessing/data_train_4.npy'

path_data_test_1 = '/home/a_santos/Documents/TEC de Monterrey/Semestre_3/Receonocimiento de patrones/Proyecto_Final/Data_preprocessing/data_test_1.npy'
path_data_test_2 = '/home/a_santos/Documents/TEC de Monterrey/Semestre_3/Receonocimiento de patrones/Proyecto_Final/Data_preprocessing/data_test_2.npy'
path_data_test_3 = '/home/a_santos/Documents/TEC de Monterrey/Semestre_3/Receonocimiento de patrones/Proyecto_Final/Data_preprocessing/data_test_3.npy'
path_data_test_4 = '/home/a_santos/Documents/TEC de Monterrey/Semestre_3/Receonocimiento de patrones/Proyecto_Final/Data_preprocessing/data_test_4.npy'

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

X_test_list = ['X_test_1', 'X_test_2', 'X_test_3', 'X_test_4']
Y_test_list = ['Y_test_1', 'Y_test_2', 'Y_test_3', 'Y_test_4']

#%% Best C value determination (C varying)

C = np.array([0.1, 0.5, 1., 5., 10.])

success_rate = np.zeros([len(C), 4])
mse = np.zeros([len(C), 4])
elapsed = np.zeros([len(C), 4])

for j in range(4):
    print('New running')
    X_train = eval(X_train_list[j])
    Y_train = eval(Y_train_list[j])
    k = 0
    for i in C:
        SVM_clf = SVC(C=i, kernel='linear')
        t = time.time()
        SVM_clf.fit(X_train, Y_train)
        print('Training ok!')
        Y_train_out = SVM_clf.predict(X_train)
        elapsed[k, j] = time.time() - t
        cnf_matrix_train = confusion_matrix(Y_train, Y_train_out)
        success_rate[k, j] = ((cnf_matrix_train[0, 0] + cnf_matrix_train[1, 1]) / \
                (cnf_matrix_train[0, 0] + cnf_matrix_train[0, 1] + \
                 cnf_matrix_train[1, 0] + cnf_matrix_train[1, 1]))
        mse[k, j] = (mean_squared_error(Y_train, Y_train_out))
        del(SVM_clf)
        print('C = ', i, '  running = ', j+1, '  time = ', elapsed[k, j])
        k = k + 1

#%% In this approach I consider that the time is very important, so I make
# a evaluation of the time in each running
f, ax = plt.subplots()
ax.plot(C, elapsed[:, 0], 'g', label='run 1')
ax.plot(C, elapsed[:, 1], 'b', label='run 2')
ax.plot(C, elapsed[:, 2], 'r', label='run 3')
ax.plot(C, elapsed[:, 3], 'm', label='run 4')
ax.set_title('Execution_time')
ax.set_xticks(C)
ax.set_xlabel('C_parameter')
ax.set_ylabel('Seconds')
ax.grid(True)

legend = ax.legend(loc='upper left')

#%% Implementation with C = 1 (It was the best)
cnf_matrix_test = np.zeros([2, 2, 4])
success_rate = np.zeros([4, 1])
mse = np.zeros([4, 1])
for i in range(4):
    print('running = ', i+1)
    X_train = eval(X_train_list[i])
    Y_train = eval(Y_train_list[i])
    X_test = eval(X_test_list[i])
    Y_test = eval(Y_test_list[i])
    SVM_clf = SVC(C=1, kernel='linear')
    t = time.time()
    SVM_clf.fit(X_train, Y_train)
    print('Training ok!')
    Y_test_out = SVM_clf.predict(X_test)
    elapsed = time.time() - t
    cnf_matrix_test[:, :, i] = confusion_matrix(Y_test, Y_test_out)
    success_rate[i] = ((cnf_matrix_test[0, 0, i] + cnf_matrix_test[1, 1, i]) / \
            (cnf_matrix_test[0, 0, i] + cnf_matrix_test[0, 1, i] + \
             cnf_matrix_test[1, 0, i] + cnf_matrix_test[1, 1, i]))
    mse[i] = (mean_squared_error(Y_test, Y_test_out))
    del(SVM_clf)
    print('time = ', elapsed)
#%%*****************************************************************************
# Confusion matrices plotting for the testing data
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
                      title='Default payments / Test data / run 3')

plt.subplot(224)
plot_confusion_matrix(cnf_matrix_test[:, :, 3].astype(int), classes=class_names,
                      title='Default payments / Test data / run 4')
