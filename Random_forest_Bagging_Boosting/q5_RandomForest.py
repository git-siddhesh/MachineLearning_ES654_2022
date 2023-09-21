import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor
from sklearn import datasets
from tree.randomForest import RandomForestClassifier
from sklearn.datasets import make_classification

# import random_forest_classification and random_forest_regression from sklearn
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier_sklearn
from sklearn.ensemble import RandomForestRegressor as RandomForestRegressor_sklearn

import matplotlib.pyplot as plt
import numpy as np

from random_forest_classification import RandomForestClassifierPlot
from random_forest_classification import RandomForestRegressorPlot

np.random.seed(42)

################### RandomForestClassifier ###################
print("ROW SAMPLING")
print('########### RandomForestClassifier ###################')
N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size = N), dtype="category")
print("X.shape: ", X.shape)
print("y.shape: ", y.shape)

for criteria in ['information_gain', 'gini_index']:
    Classifier_RF = RandomForestClassifier(10, criterion = criteria)
    Classifier_RF.fit(X, y)
    y_hat = Classifier_RF.predict(X)
    Classifier_RF.plot(criteria)
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y, print_sklearn=False))
    for cls in y.unique():
        print('class: ', cls)
        print('Precision of {} is: '.format(cls), precision(y_hat, y, cls, print_sklearn=False))
        print('Recall of {} is: '.format(cls), recall(y_hat, y, cls, print_sklearn=False))

print("--------SKLEARN--------")
for criteria in ['entropy', 'gini']:
    Classifier_RF_sklearn = RandomForestClassifier_sklearn(10, criterion = criteria)
    Classifier_RF_sklearn.fit(X, y)
    y_hat = Classifier_RF_sklearn.predict(X)
    # Classifier_RF.plot()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y, print_sklearn=False))
    for cls in y.unique():
        print('class: ', cls)
        print('Precision of {} is: '.format(cls), precision(y_hat, y, cls, print_sklearn=False))
        print('Recall of {} is: '.format(cls), recall(y_hat, y, cls, print_sklearn=False))
###################### RandomForestRegressor ###################
print('########### RandomForestRegressor ###################')
N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
print("X.shape: ", X.shape)
print("y.shape: ", y.shape)

Regressor_RF = RandomForestRegressor(10)
Regressor_RF.fit(X, y)
y_hat = Regressor_RF.predict(X)
Regressor_RF.plot()
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
print("regressor plot")
# Regressor_RF.plot()


print("--------SKLEARN--------")
Regressor_RF_sklearn = RandomForestRegressor_sklearn(10)
Regressor_RF_sklearn.fit(X, y)
y_hat = Regressor_RF_sklearn.predict(X)
# Regressor_RF.plot()
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))



#############################################################################################
# ################### RandomForestClassifier ###################
# print("COLUMN SAMPLING")
# print('########### RandomForestClassifier ###################')
# N = 30
# P = 5
# X = pd.DataFrame(np.random.randn(N, P))
# y = pd.Series(np.random.randint(P, size = N), dtype="category")
# print("X.shape: ", X.shape)
# print("y.shape: ", y.shape)

# for criteria in ['information_gain', 'gini_index']:
#     Classifier_RF = RandomForestClassifierPlot(10, criterion = criteria)
#     Classifier_RF.fit(X, y)
#     y_hat = Classifier_RF.predict(X)
#     # Classifier_RF.plot()
#     print('Criteria :', criteria)
#     print('Accuracy: ', accuracy(y_hat, y, print_sklearn=False))
#     for cls in y.unique():
#         print('class: ', cls)
#         print('Precision of {} is: '.format(cls), precision(y_hat, y, cls, print_sklearn=False))
#         print('Recall of {} is: '.format(cls), recall(y_hat, y, cls, print_sklearn=False))
#     Classifier_RF.plot()
#     Classifier_RF.plot_surface(X,y)

# print("--------SKLEARN--------")
# for criteria in ['entropy', 'gini']:
#     Classifier_RF_sklearn = RandomForestClassifier_sklearn(10, criterion = criteria)
#     Classifier_RF_sklearn.fit(X, y)
#     y_hat = Classifier_RF_sklearn.predict(X)
#     # Classifier_RF.plot()
#     print('Criteria :', criteria)
#     print('Accuracy: ', accuracy(y_hat, y, print_sklearn=False))
#     for cls in y.unique():
#         print('class: ', cls)
#         print('Precision of {} is: '.format(cls), precision(y_hat, y, cls, print_sklearn=False))
#         print('Recall of {} is: '.format(cls), recall(y_hat, y, cls, print_sklearn=False))
# ###################### RandomForestRegressor ###################
# print('########### RandomForestRegressor ###################')
# N = 30
# P = 5
# X = pd.DataFrame(np.random.randn(N, P))
# y = pd.Series(np.random.randn(N))
# print("X.shape: ", X.shape)
# print("y.shape: ", y.shape)

# Regressor_RF = RandomForestRegressorPlot(10)
# Regressor_RF.fit(X, y)
# y_hat = Regressor_RF.predict(X)
# # Regressor_RF.plot()
# print('RMSE: ', rmse(y_hat, y))
# print('MAE: ', mae(y_hat, y))
# Regressor_RF.plot()
# Regressor_RF.plot_surface(X,y)

# print("--------SKLEARN--------")
# Regressor_RF_sklearn = RandomForestRegressor_sklearn(10)
# Regressor_RF_sklearn.fit(X, y)
# y_hat = Regressor_RF_sklearn.predict(X)
# # Regressor_RF.plot()
# print('RMSE: ', rmse(y_hat, y))
# print('MAE: ', mae(y_hat, y))



################## 5 (b) #####################

N = 100
P = 10
NUM_OP_CLASSES = 2
n_estimators = 10
#X = pd.DataFrame(np.abs(np.random.randn(N, P)))
#y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")
X, y = make_classification(n_samples=N, n_features=P, n_informative=P, n_redundant=0, random_state=42, n_classes=NUM_OP_CLASSES)
# X = np.abs(np.random.randn(N, P))
# y = np.random.randint(NUM_OP_CLASSES, size=N)

# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train a random forest classifier
clf = RandomForestClassifierPlot(n_estimators=n_estimators, max_depth=5, max_features=2)
clf.fit(X, y)

# Test the random forest classifier
y_pred = clf.predict(X)
# accuracy = accuracy(y, y_pred, print_sklearn=False)

# # Print the accuracy of the random forest classifier
# print("Accuracy:", accuracy)
print("classification plot")
clf.plot()
