# -*- coding: utf-8 -*-
"""Q1_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qS8-5c3lwMfeY1LGlMIEVk-f-5K2bDZC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linear_regression import LinearRegression
from metrics import *
import time

np.random.seed(45)

N = 300
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
print(X.shape)

# compare time taken for each method


#Evaluating sklearn's implementation of linear regression
time_list = []
min_rmse = 100000000
min_mae = 100000000

for i in range(10):
    time1 = time.time()
    LR = LinearRegression(fit_intercept=True)
    LR.fit_sklearn_LR(X,y)
    time_list.append(time.time()-time1)
    min_rmse = min(min_rmse, rmse(LR.predict(X), y))
    min_mae = min(min_mae, mae(LR.predict(X), y))

print('For linear regression using sklearn : \n')
print('RMSE: ', min_rmse)
print('MAE: ', min_mae)
print("Time taken: ", sum(time_list)/len(time_list))
print("---------------------------")


#Evaluating solution of linear regression using normal equations
time_list = []
min_rmse = 100000000
min_mae = 100000000

for i in range(10):
    time1 = time.time()
    LR = LinearRegression(fit_intercept=True)
    LR.fit_normal_equations(X,y)
    time_list.append(time.time()-time1)
    min_rmse = min(min_rmse, rmse(LR.predict(X), y))
    min_mae = min(min_mae, mae(LR.predict(X), y))

print('For linear regression using normal equations : \n')
print('RMSE: ', min_rmse)
print('MAE: ', min_mae)
print("Time taken: ", sum(time_list)/len(time_list))
print("---------------------------")





#Evaluating solution of linear regression using SVD
time_list = []
min_rmse = 100000000
min_mae = 100000000

for i in range(10):
    time1 = time.time()
    LR = LinearRegression(fit_intercept=True)
    LR.fit_SVD(X,y)
    time_list.append(time.time()-time1)
    min_rmse = min(min_rmse, rmse(LR.predict(X), y))
    min_mae = min(min_mae, mae(LR.predict(X), y))

print('For linear regression using SVD : \n')
print('RMSE: ', min_rmse)
print('MAE: ', min_mae)
print("Time taken: ", sum(time_list)/len(time_list))
print("---------------------------")

# time1 = time.time()
# LR = LinearRegression(fit_intercept=True)
# LR.fit_SVD(X,y)
# print("Time taken for linear regression using SVD: ", time.time()-time1)
# y_hat = LR.predict(X)

# print('For linear regression using SVD : \n')
# print('RMSE: ', rmse(y_hat, y))
# print('MAE: ', mae(y_hat, y))
# print("---------------------------")