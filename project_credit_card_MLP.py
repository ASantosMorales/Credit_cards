#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 19:48:57 2017

@author: a_santos

Solving the Default payment of credit cards problem with Neural Networks.

* Here is a systematic evaluation to find the best neural network architecture
  to predict the default payment of credit cards.
* This script needs the preprocessed data obtained on the 
  project_credit_card_splitdata.py script.
* The best architecture will be the one with 55 neurons in the hidden layer.

"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.neural_network import MLPClassifier
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
mse_train_list = ['mse_train_1', 'mse_train_2', 'mse_train_3', 'mse_train_4']
success_rate_train_list = ['success_rate_train_1', 'success_rate_train_2', 'success_rate_train_3', 'success_rate_train_4']

X_test_list = ['X_test_1', 'X_test_2', 'X_test_3', 'X_test_4']
Y_test_list = ['Y_test_1', 'Y_test_2', 'Y_test_3', 'Y_test_4']

#%% Best architecture determination. Each architecture will be tested 10 times
# to assess its performance from 5 to 100 neurons in the hidden layer.
trials = 10
neurons = np.arange(5, 105, 5)

for l in range(4):
    print('Running = ', l)
    X_train = eval(X_train_list[l])
    Y_train = eval(Y_train_list[l])
    success_rate = np.zeros([len(neurons), trials])
    mse = np.zeros([len(neurons), trials])
    for k in range(trials):
        j = 0
        for i in neurons:
            MLP = MLPClassifier(hidden_layer_sizes=(i,), activation='relu', solver='sgd')
            MLP.fit(X_train, Y_train)
            Y_train_out = MLP.predict(X_train)
            cnf_matrix_train = confusion_matrix(Y_train, Y_train_out)
            success_rate[j, k] = ((cnf_matrix_train[0, 0] + cnf_matrix_train[1, 1]) / \
                        (cnf_matrix_train[0, 0] + cnf_matrix_train[0, 1] + \
                         cnf_matrix_train[1, 0] + cnf_matrix_train[1, 1]))
            mse[j, k] = (mean_squared_error(Y_train, Y_train_out))
            print('Trial = ', k + 1, '   Neurons = ', i)
            j = j + 1
            del(MLP)
    globals()[success_rate_train_list[l]] = success_rate
    globals()[mse_train_list[l]] = mse
            
# Plotting of the MSE of each running
for i in range(4):
    mse = eval(mse_train_list[i])
    plt.figure(i)
    plt.subplot(211)
    plt.boxplot(mse.T, labels = ['5', '10', '15', 
                                 '20', '25', '30', 
                                 '35', '40', '45', 
                                 '50', '55', '60', 
                                 '65', '70', '75', 
                                 '80', '85', '90', 
                                 '95', '100'])
    plt.title('Mean_Squared_Error along Neurons_number')
    plt.xlabel('Neurons')
    plt.ylabel('MSE')
    plt.grid(True)
    
    success_rate = eval(success_rate_train_list[i])
    plt.subplot(212)
    plt.boxplot(success_rate.T, labels = ['5', '10', '15', 
                                          '20', '25', '30', 
                                          '35', '40', '45', 
                                          '50', '55', '60', 
                                          '65', '70', '75', 
                                          '80', '85', '90', 
                                          '95', '100'])
    plt.title('Success_rate along Neurons_number')
    plt.xlabel('Neurons')
    plt.ylabel('Success rate')
    plt.grid(True)

#%%****************************************************************************
# The one with 55 neurons in the hidden layer is chosen as the best architecture
# This architecture will be tested 10 times to report its performance

trials = 10
cnf_matrix_test = np.zeros([2, 2, 4, 10])
mse_test = np.zeros([trials, 4])
success_rate_test = np.zeros([trials, 4])
    
for j in range(4):
    X_train = eval(X_train_list[j])
    Y_train = eval(Y_train_list[j])
    X_test = eval(X_test_list[j])
    Y_test = eval(Y_test_list[j])
    for k in range(trials):
        MLP = MLPClassifier(hidden_layer_sizes=55, activation='relu', solver='sgd')
        MLP.fit(X_train, Y_train)
        Y_test_out = MLP.predict(X_test)
        cnf_matrix_test[:, :, j, k] = confusion_matrix(Y_test, Y_test_out)
        np.set_printoptions(precision=2)
        success_rate_test[k, j] = ((cnf_matrix_test[0, 0, j, k] + cnf_matrix_test[1, 1, j, k]) / \
                    (cnf_matrix_test[0, 0, j, k] + cnf_matrix_test[0, 1, j, k] + \
                     cnf_matrix_test[1, 0, j, k] + cnf_matrix_test[1, 1, j, k]))
        mse_test[k, j] = (mean_squared_error(Y_test, Y_test_out))
        print(k + 1)
        print(j + 1)
        del(MLP)

# Ploteo de análisis
plt.figure()
plt.subplot(211)
plt.boxplot(mse_test, labels = ['run_1', 'run_2', 'run_3', 'run_4'])
plt.title('Mean_Squared_Error along Runnings')
plt.xlabel('Running')
plt.ylabel('MSE')
plt.grid(True)
    
plt.subplot(212)
plt.boxplot(success_rate_test, labels = ['run_1', 'run_2', 'run_3', 'run_4'])
plt.title('Success_rate along Runnings')
plt.xlabel('Running')
plt.ylabel('Success rate')
plt.grid(True)

#%% Confusion matrix plotting of the best behavior
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


plt.figure()
plt.subplot(221)
plot_confusion_matrix(cnf_matrix_test[:, :, 0, 2].astype(int), classes=class_names,
                      title='Default payments / Test data / run 1')

plt.subplot(222)
plot_confusion_matrix(cnf_matrix_test[:, :, 1, 9].astype(int), classes=class_names,
                      title='Default payments / Test data / run 2')

plt.subplot(223)
plot_confusion_matrix(cnf_matrix_test[:, :, 2, 4].astype(int), classes=class_names,
                      title='Default payments / Test data / run 2')

plt.subplot(224)
plot_confusion_matrix(cnf_matrix_test[:, :, 3, 4].astype(int), classes=class_names,
                      title='Default payments / Test data / run 2')