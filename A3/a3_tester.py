# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 22:17:44 2022

@author: richm
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from logistic_regression import fit, predict


     
        
# import the data file
df = pd.read_csv("engine_condition.csv")       # linearly separable
# df = pd.read_csv("engine_condition_comp.csv")    # not linearly separable
feature_names = ['EV1', 'EV2']
X_all = df[feature_names]
y_all = df['Status']

# split into training and test sets for holdout validation
# cross validation would be a better approach with such a small dataset
# converting from DataFrame to numpy arrays in the process
X_train, X_test, y_train, y_test = train_test_split(X_all.to_numpy(), y_all.to_numpy(), random_state=42)

# train the model and display the learned weights
eta=0.1
n_epoch=100
weights = fit(X_train, y_train, eta, n_epoch)
print(f'LEARNED WEIGHTS :\n\t{weights}')


# generate evaluation statistics for training and test set
N_train = X_train.shape[0]
N_right = 0
N_wrong = 0
for i in range(N_train):
    
    if abs(y_train[i]-predict(X_train[i,:],weights))>0.0:
        N_wrong += 1
    else:
        N_right += 1
train_accuracy = N_right/N_train
    

print('\nACCURACY STATISTICS')
print('On training set:')
print(f'\t   total cases = {N_train:d}')
print(f'\t       correctly classified   = {N_right:d}')
print(f'\t       incorrectly classified = {N_wrong:d}')
print(f'\taccuracy = {100*train_accuracy:4.2f} %')
    

N_test = X_test.shape[0]
N_right = 0
N_wrong = 0
for i in range(N_test):
    if abs(y_test[i]-predict(X_test[i,:],weights))>0.0:
        N_wrong += 1
    else:
        N_right += 1
test_accuracy = N_right/N_test
    
print('On test set:')
print(f'\t   total cases = {N_test:d}')
print(f'\t       correctly classified   = {N_right:d}')
print(f'\t       incorrectly classified = {N_wrong:d}')
print(f'\taccuracy = {100*test_accuracy:4.2f} %')
    


# plot the data and learned boundary
# for simplicity, we are not distinguishing between training and test data 

# this step calculates points along the classification boundary
# will fail if boundary is vertical (weights[2]==0)
x1 = np.linspace(4.4, 6.3, num=2)
x2 = - weights[0]/weights[2] - weights[1]/weights[2]*x1

# separate data into class 1 and class 0
# inefficient for very large sets (due to appending) but fine for us here
indices1 = []
indices0 = []
for i in range(N_train+N_test):
    if y_all[i] == 1.0:
        indices1.append(i)
    else:
        indices0.append(i)

df_arr = np.array(df)
D0 = df_arr[indices0]
D1 = df_arr[indices1]

plt.figure()
plt.scatter(D1[:, 0], D1[:, 1], marker='o', s=50, edgecolor='k', label='Good')
plt.scatter(D0[:, 0], D0[:, 1], marker='s', s=50, edgecolor='k', label='Diminished')
plt.plot(x1, x2, label='Decision Boundary')
plt.xlabel("Engine Variable 1")
plt.ylabel("Engine Variable 2")
plt.legend()




