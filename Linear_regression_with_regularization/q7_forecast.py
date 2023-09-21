# #TODO : Write here
# # -*- coding: utf-8 -*-
# """Q7_forecast.ipynb

# Automatically generated by Colaboratory.

# Original file is located at
#     https://colab.research.google.com/drive/1QygJyzTuf4wJ3ioxCqF0boFIbPNbBJd6
# """
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt

# data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv', header=0, parse_dates=[0], index_col=0).squeeze()

# split_date = pd.to_datetime('1988-01-01')  # Set the date to split the data into training and testing sets

# train_data = data[:split_date]
# test_data = data[split_date:]

# X_train = pd.DataFrame({'t': train_data[:-1]})
# y_train = train_data[1:]

# X_test = pd.DataFrame({'t': test_data[:-1]})
# y_test = test_data[1:]

# lr = LinearRegression()
# lr.fit(X_train, y_train)

# y_pred_train = lr.predict(X_train)
# y_pred_test = lr.predict(X_test)

# rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
# rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)

# print('Train RMSE:', rmse_train)
# print('Test RMSE:', rmse_test)

# plt.figure(figsize=(12, 6))
# plt.plot(train_data.index[1:], y_train, label='True Train', color='#e4e4c5')
# plt.plot(train_data.index[1:], y_pred_train, label='Predicted Train', color='#9bc1bc')
# plt.title('Train Data')
# plt.legend()
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.plot(test_data.index[1:], y_test, label='True Test', color='#e4e499')
# plt.plot(test_data.index[1:], y_pred_test, label='Predicted Test', color='#5ca4a9')
# plt.title('Test Data')
# plt.legend()
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.plot(train_data.index[1:], y_train, label='True Train', color='#e4e4c5')
# plt.plot(train_data.index[1:], y_pred_train, label='Predicted Train', color='#9bc1bc')
# plt.plot(test_data.index[1:], y_test, label='True Test', color='#e4e499')
# plt.plot(test_data.index[1:], y_pred_test, label='Predicted Test', color='#5ca4a9')
# plt.title('Prediction for Train and Test Data')
# plt.axvline(x=split_date, linestyle='--', color='black', label='Split Date')
# plt.legend()
# plt.show()

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

if not os.path.exists('Plots/Question7/'):
    os.makedirs('Plots/Question7/')


# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv', header=0, index_col=0)

split_date = '1990-01-05'
split_index = df.index.get_loc(split_date)

# Create lagged variables
lags = 30
for i in range(1, lags+1):
    df[f't-{i}'] = df['Temp'].shift(i)

# Remove missing values
df.dropna(inplace=True)
print(df.head())

# Split into train and test sets
# train_size = int(len(df) * 0.8)
train, test = df.iloc[:split_index], df.iloc[split_index:]

# Fit linear regression model
X_train, y_train = train.iloc[:, 1:], train['Temp']
X_test, y_test = test.iloc[:, 1:], test['Temp']
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions on train set
y_pred_train = model.predict(X_train)

# Make predictions on test set
y_pred = model.predict(X_test)


# Calculate RMSE
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
print('Train RMSE:', rmse_train)
rmse_test = mean_squared_error(y_test, y_pred, squared=False)
print('Test RMSE:', rmse_test)

# Plot predictions vs true values
plt.plot(y_train.values, label='true')
plt.plot(y_pred_train, label='predicted')
plt.title('Train Data')
plt.legend()
plt.savefig('Plots/Question7/Q7_plot_train.png')
plt.show()
# save plot to file

plt.plot(y_test.values, label='true')
plt.plot(y_pred, label='predicted')
plt.title('Test Data')
plt.legend()
plt.savefig('Plots/Question7/Q7_plot_test.png')
plt.show()
# save plot to file




fig, ax = plt.subplots()
ax.plot(y_train, label='Train Data')
ax.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, label='Test Data')
ax.plot(y_pred_train, label='Train Data')
ax.plot(range(len(y_train), len(y_train) + len(y_test)), y_pred,  label='Test Data')
ax.set_xlabel('Index')
ax.set_ylabel('Target Values')
ax.set_title('Train and Test Data')
ax.legend()
plt.savefig('Plots/Question7/Q7_plot_train_test.png')
plt.show()