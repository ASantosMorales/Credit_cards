#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:57:57 2017

@author: a_santos

Multidimensional linear regression

y = beta_0 + beta_1*x_1 + beta_2*x_2 + ... + beta_n*x_n 

"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

#%% Get pre-processed data
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

del(path_data_train_1)
del(path_data_train_2)
del(path_data_train_3)
del(path_data_train_4)

del(path_data_test_1)
del(path_data_test_2)
del(path_data_test_3)
del(path_data_test_4)

#%% Data homogenization
data_train_1 = np.concatenate((np.ones(len(data_train_1)).reshape(len(data_train_1), 1), data_train_1), axis=1)
data_train_2 = np.concatenate((np.ones(len(data_train_2)).reshape(len(data_train_2), 1), data_train_2), axis=1)
data_train_3 = np.concatenate((np.ones(len(data_train_3)).reshape(len(data_train_3), 1), data_train_3), axis=1)
data_train_4 = np.concatenate((np.ones(len(data_train_4)).reshape(len(data_train_4), 1), data_train_4), axis=1)

data_test_1 = np.concatenate((np.ones(len(data_test_1)).reshape(len(data_test_1), 1), data_test_1), axis=1)
data_test_2 = np.concatenate((np.ones(len(data_test_2)).reshape(len(data_test_2), 1), data_test_2), axis=1)
data_test_3 = np.concatenate((np.ones(len(data_test_3)).reshape(len(data_test_3), 1), data_test_3), axis=1)
data_test_4 = np.concatenate((np.ones(len(data_test_4)).reshape(len(data_test_4), 1), data_test_4), axis=1)

#%% Beta getting
beta_1 = np.matmul(np.matmul(np.linalg.inv(np.matmul(data_train_1[:, 0:24].T, data_train_1[:, 0:24])), data_train_1[:, 0:24].T), data_train_1[:, 24])
beta_2 = np.matmul(np.matmul(np.linalg.inv(np.matmul(data_train_2[:, 0:24].T, data_train_2[:, 0:24])), data_train_2[:, 0:24].T), data_train_2[:, 24])
beta_3 = np.matmul(np.matmul(np.linalg.inv(np.matmul(data_train_3[:, 0:24].T, data_train_3[:, 0:24])), data_train_3[:, 0:24].T), data_train_3[:, 24])
beta_4 = np.matmul(np.matmul(np.linalg.inv(np.matmul(data_train_4[:, 0:24].T, data_train_4[:, 0:24])), data_train_4[:, 0:24].T), data_train_4[:, 24])

#%% Get predictions for each splitting set
y_train_1 = beta_1[0] + beta_1[1]*data_train_1[:, 1] + beta_1[2]*data_train_1[:, 2] + \
            beta_1[3]*data_train_1[:, 3] + beta_1[4]*data_train_1[:, 4] + \
            beta_1[5]*data_train_1[:, 5] + beta_1[6]*data_train_1[:, 6] + \
            beta_1[7]*data_train_1[:, 7] + beta_1[8]*data_train_1[:, 8] + \
            beta_1[9]*data_train_1[:, 9] + beta_1[10]*data_train_1[:, 10] + \
            beta_1[11]*data_train_1[:, 11] + beta_1[12]*data_train_1[:, 12] + \
            beta_1[13]*data_train_1[:, 13] + beta_1[14]*data_train_1[:, 14] + \
            beta_1[15]*data_train_1[:, 15] + beta_1[16]*data_train_1[:, 16] + \
            beta_1[17]*data_train_1[:, 17] + beta_1[18]*data_train_1[:, 18] + \
            beta_1[19]*data_train_1[:, 19] + beta_1[20]*data_train_1[:, 20] + \
            beta_1[21]*data_train_1[:, 21] + beta_1[22]*data_train_1[:, 22] + \
            beta_1[23]*data_train_1[:, 23]
            
y_train_2 = beta_2[0] + beta_2[1]*data_train_2[:, 1] + beta_2[2]*data_train_2[:, 2] + \
            beta_2[3]*data_train_2[:, 3] + beta_2[4]*data_train_2[:, 4] + \
            beta_2[5]*data_train_2[:, 5] + beta_2[6]*data_train_2[:, 6] + \
            beta_2[7]*data_train_2[:, 7] + beta_2[8]*data_train_2[:, 8] + \
            beta_2[9]*data_train_2[:, 9] + beta_2[10]*data_train_2[:, 10] + \
            beta_2[11]*data_train_2[:, 11] + beta_2[12]*data_train_2[:, 12] + \
            beta_2[13]*data_train_2[:, 13] + beta_2[14]*data_train_2[:, 14] + \
            beta_2[15]*data_train_2[:, 15] + beta_2[16]*data_train_2[:, 16] + \
            beta_2[17]*data_train_2[:, 17] + beta_2[18]*data_train_2[:, 18] + \
            beta_2[19]*data_train_2[:, 19] + beta_2[20]*data_train_2[:, 20] + \
            beta_2[21]*data_train_2[:, 21] + beta_2[22]*data_train_2[:, 22] + \
            beta_2[23]*data_train_2[:, 23]

y_train_3 = beta_3[0] + beta_3[1]*data_train_3[:, 1] + beta_3[2]*data_train_3[:, 2] + \
            beta_3[3]*data_train_3[:, 3] + beta_3[4]*data_train_3[:, 4] + \
            beta_3[5]*data_train_3[:, 5] + beta_3[6]*data_train_3[:, 6] + \
            beta_3[7]*data_train_3[:, 7] + beta_3[8]*data_train_3[:, 8] + \
            beta_3[9]*data_train_3[:, 9] + beta_3[10]*data_train_3[:, 10] + \
            beta_3[11]*data_train_3[:, 11] + beta_3[12]*data_train_3[:, 12] + \
            beta_3[13]*data_train_3[:, 13] + beta_3[14]*data_train_3[:, 14] + \
            beta_3[15]*data_train_3[:, 15] + beta_3[16]*data_train_3[:, 16] + \
            beta_3[17]*data_train_3[:, 17] + beta_3[18]*data_train_3[:, 18] + \
            beta_3[19]*data_train_3[:, 19] + beta_3[20]*data_train_3[:, 20] + \
            beta_3[21]*data_train_3[:, 21] + beta_3[22]*data_train_3[:, 22] + \
            beta_3[23]*data_train_3[:, 23]
            
y_train_4 = beta_4[0] + beta_4[1]*data_train_4[:, 1] + beta_4[2]*data_train_4[:, 2] + \
            beta_4[3]*data_train_4[:, 3] + beta_4[4]*data_train_4[:, 4] + \
            beta_4[5]*data_train_4[:, 5] + beta_4[6]*data_train_4[:, 6] + \
            beta_4[7]*data_train_4[:, 7] + beta_4[8]*data_train_4[:, 8] + \
            beta_4[9]*data_train_4[:, 9] + beta_4[10]*data_train_4[:, 10] + \
            beta_4[11]*data_train_4[:, 11] + beta_4[12]*data_train_4[:, 12] + \
            beta_4[13]*data_train_4[:, 13] + beta_4[14]*data_train_4[:, 14] + \
            beta_4[15]*data_train_4[:, 15] + beta_4[16]*data_train_4[:, 16] + \
            beta_4[17]*data_train_4[:, 17] + beta_4[18]*data_train_4[:, 18] + \
            beta_4[19]*data_train_4[:, 19] + beta_4[20]*data_train_4[:, 20] + \
            beta_4[21]*data_train_4[:, 21] + beta_4[22]*data_train_4[:, 22] + \
            beta_4[23]*data_train_4[:, 23]

#%% Threshold determination
trials = np.arange(0, 1, 0.01)

y_train_out_1 = np.zeros([len(data_train_1), len(trials)])
mse_train_1 = []

for i in range(len(trials)):
    for j in range(len(data_train_1)):
        if y_train_1[j] >= trials[i]:
            y_train_out_1[j, i] = 1
        else:
            y_train_out_1[j, i] = 0
    mse_train_1.append(mean_squared_error(data_train_1[:, 24], y_train_out_1[:, i]))

y_train_out_2 = np.zeros([len(data_train_2), len(trials)])
mse_train_2 = []

for i in range(len(trials)):
    for j in range(len(data_train_2)):
        if y_train_2[j] >= trials[i]:
            y_train_out_2[j, i] = 1
        else:
            y_train_out_2[j, i] = 0
    mse_train_2.append(mean_squared_error(data_train_2[:, 24], y_train_out_2[:, i]))

y_train_out_3 = np.zeros([len(data_train_3), len(trials)])
mse_train_3 = []

for i in range(len(trials)):
    for j in range(len(data_train_3)):
        if y_train_3[j] >= trials[i]:
            y_train_out_3[j, i] = 1
        else:
            y_train_out_3[j, i] = 0
    mse_train_3.append(mean_squared_error(data_train_3[:, 24], y_train_out_3[:, i]))

y_train_out_4 = np.zeros([len(data_train_4), len(trials)])
mse_train_4 = []
for i in range(len(trials)):
    for j in range(len(data_train_4)):
        if y_train_4[j] >= trials[i]:
            y_train_out_4[j, i] = 1
        else:
            y_train_out_4[j, i] = 0
    mse_train_4.append(mean_squared_error(data_train_4[:, 24], y_train_out_4[:, i]))



# ******* Plot trials ****************
plt.figure()
plt.subplot(221)
plt.plot(trials, mse_train_1, 'k.', trials[mse_train_1.index(min(mse_train_1))], min(mse_train_1), 'r.')
plt.title('Running 1')
plt.ylabel('Mean squared error')
plt.xlabel('Threshold')
plt.xticks(np.arange(0, 1.2, 0.2))
plt.grid(True)

plt.subplot(222)
plt.plot(trials, mse_train_2, 'k.', trials[mse_train_2.index(min(mse_train_2))], min(mse_train_2), 'r.')
plt.title('Running 2')
plt.ylabel('Mean squared error')
plt.xlabel('Threshold')
plt.xticks(np.arange(0, 1.2, 0.2))
plt.grid(True)

plt.subplot(223)
plt.plot(trials, mse_train_3, 'k.', trials[mse_train_3.index(min(mse_train_3))], min(mse_train_3), 'r.')
plt.title('Running 3')
plt.ylabel('Mean squared error')
plt.xlabel('Threshold')
plt.xticks(np.arange(0, 1.2, 0.2))
plt.grid(True)

plt.subplot(224)
plt.plot(trials, mse_train_4, 'k.', trials[mse_train_4.index(min(mse_train_4))], min(mse_train_4), 'r.')
plt.title('Running 4')
plt.ylabel('Mean squared error')
plt.xlabel('Threshold')
plt.xticks(np.arange(0, 1.2, 0.2))
plt.grid(True)

#%% Plotting confusion matrices to the training data


# Function to obtain a confusion matrix
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

# Compute confusion matrix
cnf_matrix_train_1 = confusion_matrix(data_train_1[:, 24], y_train_out_1[:, 40])
np.set_printoptions(precision=2)

cnf_matrix_train_2 = confusion_matrix(data_train_2[:, 24], y_train_out_2[:, 40])
np.set_printoptions(precision=2)

cnf_matrix_train_3 = confusion_matrix(data_train_3[:, 24], y_train_out_3[:, 40])
np.set_printoptions(precision=2)

cnf_matrix_train_4 = confusion_matrix(data_train_4[:, 24], y_train_out_4[:, 40])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plt.subplot(221)
plot_confusion_matrix(cnf_matrix_train_1, classes=class_names,
                      title='Default payments / Train data / run 1')

plt.subplot(222)
plot_confusion_matrix(cnf_matrix_train_2, classes=class_names,
                      title='Default payments / Train data / run 2')

plt.subplot(223)
plot_confusion_matrix(cnf_matrix_train_3, classes=class_names,
                      title='Default payments / Train data / run 3')

plt.subplot(224)
plot_confusion_matrix(cnf_matrix_train_4, classes=class_names,
                      title='Default payments / Train data / run 4')

#%% Y determination for the testing data
y_test_1 = beta_1[0] + beta_1[1]*data_test_1[:, 1] + beta_1[2]*data_test_1[:, 2] + \
            beta_1[3]*data_test_1[:, 3] + beta_1[4]*data_test_1[:, 4] + \
            beta_1[5]*data_test_1[:, 5] + beta_1[6]*data_test_1[:, 6] + \
            beta_1[7]*data_test_1[:, 7] + beta_1[8]*data_test_1[:, 8] + \
            beta_1[9]*data_test_1[:, 9] + beta_1[10]*data_test_1[:, 10] + \
            beta_1[11]*data_test_1[:, 11] + beta_1[12]*data_test_1[:, 12] + \
            beta_1[13]*data_test_1[:, 13] + beta_1[14]*data_test_1[:, 14] + \
            beta_1[15]*data_test_1[:, 15] + beta_1[16]*data_test_1[:, 16] + \
            beta_1[17]*data_test_1[:, 17] + beta_1[18]*data_test_1[:, 18] + \
            beta_1[19]*data_test_1[:, 19] + beta_1[20]*data_test_1[:, 20] + \
            beta_1[21]*data_test_1[:, 21] + beta_1[22]*data_test_1[:, 22] + \
            beta_1[23]*data_test_1[:, 23]
            
y_test_2 = beta_2[0] + beta_2[1]*data_test_2[:, 1] + beta_2[2]*data_test_2[:, 2] + \
            beta_2[3]*data_test_2[:, 3] + beta_2[4]*data_test_2[:, 4] + \
            beta_2[5]*data_test_2[:, 5] + beta_2[6]*data_test_2[:, 6] + \
            beta_2[7]*data_test_2[:, 7] + beta_2[8]*data_test_2[:, 8] + \
            beta_2[9]*data_test_2[:, 9] + beta_2[10]*data_test_2[:, 10] + \
            beta_2[11]*data_test_2[:, 11] + beta_2[12]*data_test_2[:, 12] + \
            beta_2[13]*data_test_2[:, 13] + beta_2[14]*data_test_2[:, 14] + \
            beta_2[15]*data_test_2[:, 15] + beta_2[16]*data_test_2[:, 16] + \
            beta_2[17]*data_test_2[:, 17] + beta_2[18]*data_test_2[:, 18] + \
            beta_2[19]*data_test_2[:, 19] + beta_2[20]*data_test_2[:, 20] + \
            beta_2[21]*data_test_2[:, 21] + beta_2[22]*data_test_2[:, 22] + \
            beta_2[23]*data_test_2[:, 23]

y_test_3 = beta_3[0] + beta_3[1]*data_test_3[:, 1] + beta_3[2]*data_test_3[:, 2] + \
            beta_3[3]*data_test_3[:, 3] + beta_3[4]*data_test_3[:, 4] + \
            beta_3[5]*data_test_3[:, 5] + beta_3[6]*data_test_3[:, 6] + \
            beta_3[7]*data_test_3[:, 7] + beta_3[8]*data_test_3[:, 8] + \
            beta_3[9]*data_test_3[:, 9] + beta_3[10]*data_test_3[:, 10] + \
            beta_3[11]*data_test_3[:, 11] + beta_3[12]*data_test_3[:, 12] + \
            beta_3[13]*data_test_3[:, 13] + beta_3[14]*data_test_3[:, 14] + \
            beta_3[15]*data_test_3[:, 15] + beta_3[16]*data_test_3[:, 16] + \
            beta_3[17]*data_test_3[:, 17] + beta_3[18]*data_test_3[:, 18] + \
            beta_3[19]*data_test_3[:, 19] + beta_3[20]*data_test_3[:, 20] + \
            beta_3[21]*data_test_3[:, 21] + beta_3[22]*data_test_3[:, 22] + \
            beta_3[23]*data_test_3[:, 23]
            
y_test_4 = beta_4[0] + beta_4[1]*data_test_4[:, 1] + beta_4[2]*data_test_4[:, 2] + \
            beta_4[3]*data_test_4[:, 3] + beta_4[4]*data_test_4[:, 4] + \
            beta_4[5]*data_test_4[:, 5] + beta_4[6]*data_test_4[:, 6] + \
            beta_4[7]*data_test_4[:, 7] + beta_4[8]*data_test_4[:, 8] + \
            beta_4[9]*data_test_4[:, 9] + beta_4[10]*data_test_4[:, 10] + \
            beta_4[11]*data_test_4[:, 11] + beta_4[12]*data_test_4[:, 12] + \
            beta_4[13]*data_test_4[:, 13] + beta_4[14]*data_test_4[:, 14] + \
            beta_4[15]*data_test_4[:, 15] + beta_4[16]*data_test_4[:, 16] + \
            beta_4[17]*data_test_4[:, 17] + beta_4[18]*data_test_4[:, 18] + \
            beta_4[19]*data_test_4[:, 19] + beta_4[20]*data_test_4[:, 20] + \
            beta_4[21]*data_test_4[:, 21] + beta_4[22]*data_test_4[:, 22] + \
            beta_4[23]*data_test_4[:, 23]

#%% Applying the same threshold to the testing data

y_test_out_1 = np.zeros([len(data_test_1), 1])
for i in range(len(data_test_1)):
        if y_test_1[i] >= 0.4:
            y_test_out_1[i] = 1
        else:
            y_test_out_1[i] = 0
mse_test_1 = mean_squared_error(data_test_1[:, 24], y_test_out_1)

y_test_out_2 = np.zeros([len(data_test_2), 1])
for i in range(len(data_test_2)):
        if y_test_2[i] >= 0.4:
            y_test_out_2[i] = 1
        else:
            y_test_out_2[i] = 0
mse_test_2 = mean_squared_error(data_test_2[:, 24], y_test_out_2)

y_test_out_3 = np.zeros([len(data_test_3), 1])
for i in range(len(data_test_3)):
        if y_test_3[i] >= 0.4:
            y_test_out_3[i] = 1
        else:
            y_test_out_3[i] = 0
mse_test_3 = mean_squared_error(data_test_3[:, 24], y_test_out_3)

y_test_out_4 = np.zeros([len(data_test_4), 1])
for i in range(len(data_test_4)):
        if y_test_4[i] >= 0.4:
            y_test_out_4[i] = 1
        else:
            y_test_out_4[i] = 0
mse_test_4 = mean_squared_error(data_test_4[:, 24], y_test_out_4)

#%% Plotting confusion matrices to the testing data

# Compute confusion matrix
cnf_matrix_test_1 = confusion_matrix(data_test_1[:, 24], y_test_out_1)
np.set_printoptions(precision=2)

cnf_matrix_test_2 = confusion_matrix(data_test_2[:, 24], y_test_out_2)
np.set_printoptions(precision=2)

cnf_matrix_test_3 = confusion_matrix(data_test_3[:, 24], y_test_out_3)
np.set_printoptions(precision=2)

cnf_matrix_test_4 = confusion_matrix(data_test_4[:, 24], y_test_out_4)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plt.subplot(221)
plot_confusion_matrix(cnf_matrix_test_1, classes=class_names,
                      title='Default payments / Test data / run 1')

plt.subplot(222)
plot_confusion_matrix(cnf_matrix_test_2, classes=class_names,
                      title='Default payments / Test data / run 2')

plt.subplot(223)
plot_confusion_matrix(cnf_matrix_test_3, classes=class_names,
                      title='Default payments / Test data / run 3')

plt.subplot(224)
plot_confusion_matrix(cnf_matrix_test_4, classes=class_names,
                      title='Default payments / Test data / run 4')


#%% Compute the success rate for each subset
success_rate = np.zeros([4, 1])
success_rate[0] = ((cnf_matrix_test_1[0, 0] + cnf_matrix_test_1[1, 1]) / \
                (cnf_matrix_test_1[0, 0] + cnf_matrix_test_1[0, 1] + \
                 cnf_matrix_test_1[1, 0] + cnf_matrix_test_1[1, 1]))
success_rate[1] = ((cnf_matrix_test_2[0, 0] + cnf_matrix_test_2[1, 1]) / \
                (cnf_matrix_test_2[0, 0] + cnf_matrix_test_2[0, 1] + \
                 cnf_matrix_test_2[1, 0] + cnf_matrix_test_2[1, 1]))
success_rate[2] = ((cnf_matrix_test_3[0, 0] + cnf_matrix_test_3[1, 1]) / \
                (cnf_matrix_test_3[0, 0] + cnf_matrix_test_3[0, 1] + \
                 cnf_matrix_test_3[1, 0] + cnf_matrix_test_3[1, 1]))
success_rate[3] = ((cnf_matrix_test_4[0, 0] + cnf_matrix_test_4[1, 1]) / \
                (cnf_matrix_test_4[0, 0] + cnf_matrix_test_4[0, 1] + \
                 cnf_matrix_test_4[1, 0] + cnf_matrix_test_4[1, 1]))