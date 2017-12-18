#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 12:22:41 2017

@author: a_santos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick #To use scientific notation in the axis

# Get data
path = '/home/a_santos/Documents/TEC de Monterrey/Semestre_3/Receonocimiento de patrones/Proyecto_Final/default_of_credit_card_clients.xls'
xl = pd.read_excel(path)
end_data = len(xl)
data = np.array(xl.as_matrix()[1:end_data, :], dtype = np.float)
del(path)
del(end_data)
del(xl)

#%% Box diagrams of the raw data
f, axes = plt.subplots(2, 2)
#********************************************************************
axes[0, 0].boxplot(data[:, 0], labels = ['X_1'])
axes[0, 0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
axes[0, 0].set_ylabel('Dollars')
axes[0, 0].set_title('Limit balance')
axes[0, 0].grid(True)

#********************************************************************
axes[0, 1].boxplot(data[:, 4], labels = ['X_5'])
axes[0, 1].set_ylabel('Years')
axes[0, 1].set_title('Age')
axes[0, 1].grid(True)

#********************************************************************
axes[1, 0].boxplot(np.hstack((data[:, 11].reshape(30000, 1), 
    data[:, 12].reshape(30000, 1),
    data[:, 13].reshape(30000, 1),
    data[:, 14].reshape(30000, 1),
    data[:, 15].reshape(30000, 1),
    data[:, 16].reshape(30000, 1))), labels = ['X_12', 'X_13', 'X_14', 'X_15', 'X_16', 'X_17'])
axes[1, 0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
axes[1, 0].set_ylabel('Dollars')
axes[1, 0].set_title('Amount of bill statement by month')
axes[1, 0].grid(True)

#********************************************************************
axes[1, 1].boxplot(np.hstack((data[:, 17].reshape(30000, 1), 
    data[:, 18].reshape(30000, 1),
    data[:, 19].reshape(30000, 1),
    data[:, 20].reshape(30000, 1),
    data[:, 21].reshape(30000, 1),
    data[:, 22].reshape(30000, 1))), labels = ['X_18', 'X_19', 'X_20', 'X_21', 'X_22', 'X_23'])
axes[1, 1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
axes[1, 1].set_ylabel('Dollars')
axes[1, 1].set_title('Amount paid by month')
axes[1, 1].grid(True)
plt.show()

#%% Data description of the raw data
dataframe = pd.DataFrame(data, columns=['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6',
                                            'X_7', 'X_8', 'X_9', 'X_10', 'X_11', 'X_12',
                                            'X_13', 'X_14', 'X_15', 'X_16', 'X_17','X_18',
                                            'X_19', 'X_20', 'X_21', 'X_22', 'X_23', 'X_24'])
analysis = dataframe.describe()
        
#%% Histograms of the raw data
f, axes = plt.subplots(6, 4)
labels = ['Limit balance', 'Sex', 'Education',
          'Marital status', 'Age', 
          'September repayment status',
          'August repayement status', 'July repayement status',
          'June repayement status', 'May repayement status',
          'April repayement status',
          'September amount of bill statement',
          'August amount of bill statement', 'July amount of bill statement',
          'June amount of bill statement', 'May amount of bill statement',
          'April amount of bill statement',
          'Amount paid in September',
          'Amount paid in August', 'Amount paid in July',
          'Amount paid in June', 'Amount paid in May',
          'Amount paid in April',
          'Credible']

k = 0
for i in range(6):
    for j in range(4):
        axes[i, j].hist(data[:, k], bins = 'auto')
        axes[i, j].set_title(labels[k])
        if max(data[:, k]) > 100:
            axes[i, j].set_xticks(np.arange(min(data[:, k]), max(data[:, k])+1, (max(data[:, k])-min(data[:, k]))/3))
            axes[i, j].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        k = k + 1
#%% Covariance matrix of the raw data
corr_matrix = np.corrcoef(data.T)

#%% Dispersion diagrams of the data with covariance near to 1
f, axes = plt.subplots(6, 6)

for i in range(6):
    for j in range(6):
        axes[i, j].plot(data[:, i + 11], data[:, j + 11], 'k.')
        axes[i, j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
        axes[i, j].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
        axes[i, j].set_xticks(np.arange(min(data[:, i + 11]), max(data[:, i + 11])+1, (max(data[:, i + 11])-min(data[:, i + 11]))/2))

#%%***************************************************************************
#*****************************************************************************
#*****************************************************************************
# DATA MODIFICATION (escalation and logarithmic transformation to approximate a normal behavior)
data_mod = np.hstack((np.log(data[:, 0].reshape(30000, 1)),
    data[:, 1].reshape(30000, 1),
    data[:, 2].reshape(30000, 1),
    data[:, 3].reshape(30000, 1),
    np.log(data[:, 4].reshape(30000, 1)),
    data[:, 5].reshape(30000, 1),
    data[:, 6].reshape(30000, 1),
    data[:, 7].reshape(30000, 1),
    data[:, 8].reshape(30000, 1),
    data[:, 9].reshape(30000, 1),
    data[:, 10].reshape(30000, 1),
    np.log(data[:, 11].reshape(30000, 1) + (max(data[:, 11]) - min(data[:, 11]))/2),
    np.log(data[:, 12].reshape(30000, 1) + (max(data[:, 12]) - min(data[:, 12]))/2),
    np.log(data[:, 13].reshape(30000, 1) + (max(data[:, 13]) - min(data[:, 13]))/2),
    np.log(data[:, 14].reshape(30000, 1) + (max(data[:, 14]) - min(data[:, 14]))/2),
    np.log(data[:, 15].reshape(30000, 1) + (max(data[:, 15]) - min(data[:, 15]))/2),
    np.log(data[:, 16].reshape(30000, 1) + (max(data[:, 16]) - min(data[:, 16]))/2),
    np.log(data[:, 17].reshape(30000, 1) + (max(data[:, 17]) - min(data[:, 17]))/2),
    np.log(data[:, 18].reshape(30000, 1) + (max(data[:, 18]) - min(data[:, 18]))/2),
    np.log(data[:, 19].reshape(30000, 1) + (max(data[:, 19]) - min(data[:, 19]))/2),
    np.log(data[:, 20].reshape(30000, 1) + (max(data[:, 20]) - min(data[:, 20]))/2),
    np.log(data[:, 21].reshape(30000, 1) + (max(data[:, 21]) - min(data[:, 21]))/2),
    np.log(data[:, 22].reshape(30000, 1) + (max(data[:, 22]) - min(data[:, 22]))/2),
    data[:, 23].reshape(30000, 1)))

#%% Data description of the modified data
dataframe = pd.DataFrame(data_mod, columns=['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6',
                                            'X_7', 'X_8', 'X_9', 'X_10', 'X_11', 'X_12',
                                            'X_13', 'X_14', 'X_15', 'X_16', 'X_17','X_18',
                                            'X_19', 'X_20', 'X_21', 'X_22', 'X_23', 'X_24'])
analysis = dataframe.describe()

#%% Box diagrams of the modified data
f, axes = plt.subplots(2, 2)
#********************************************************************
axes[0, 0].boxplot(data_mod[:, 0], labels = ['X_1'])
axes[0, 0].set_ylabel('Dollars')
axes[0, 0].set_title('Limit balance')
axes[0, 0].grid(True)

#********************************************************************
axes[0, 1].boxplot(data_mod[:, 4], labels = ['X_5'])
axes[0, 1].set_ylabel('Years')
axes[0, 1].set_title('Age')
axes[0, 1].grid(True)

#********************************************************************
axes[1, 0].boxplot(np.hstack((data_mod[:, 11].reshape(30000, 1), 
    data_mod[:, 12].reshape(30000, 1),
    data_mod[:, 13].reshape(30000, 1),
    data_mod[:, 14].reshape(30000, 1),
    data_mod[:, 15].reshape(30000, 1),
    data_mod[:, 16].reshape(30000, 1))), labels = ['X_12', 'X_13', 'X_14', 'X_15', 'X_16', 'X_17'])
axes[1, 0].set_ylabel('Dollars')
axes[1, 0].set_title('Amount of bill statement by month')
axes[1, 0].grid(True)

#********************************************************************
axes[1, 1].boxplot(np.hstack((data_mod[:, 17].reshape(30000, 1), 
    data_mod[:, 18].reshape(30000, 1),
    data_mod[:, 19].reshape(30000, 1),
    data_mod[:, 20].reshape(30000, 1),
    data_mod[:, 21].reshape(30000, 1),
    data_mod[:, 22].reshape(30000, 1))), labels = ['X_18', 'X_19', 'X_20', 'X_21', 'X_22', 'X_23'])
axes[1, 1].set_ylabel('Dollars')
axes[1, 1].set_title('Amount paid by month')
axes[1, 1].grid(True)
plt.show()

#%% Histograms of the modified data
f, axes = plt.subplots(6, 4)
labels = ['Limit balance', 'Sex', 'Education',
          'Marital status', 'Age', 
          'September repayment status',
          'August repayement status', 'July repayement status',
          'June repayement status', 'May repayement status',
          'April repayement status',
          'September amount of bill statement',
          'August amount of bill statement', 'July amount of bill statement',
          'June amount of bill statement', 'May amount of bill statement',
          'April amount of bill statement',
          'Amount paid in September',
          'Amount paid in August', 'Amount paid in July',
          'Amount paid in June', 'Amount paid in May',
          'Amount paid in April',
          'Credible']

k = 0
for i in range(6):
    for j in range(4):
        axes[i, j].hist(data_mod[:, k], bins = 'auto')
        axes[i, j].set_title(labels[k])
        k = k + 1
#%%***************************************************************************
#****************************Saving the modified data*************************
#*****************************************************************************

np.save('data_modified.npy', data_mod)