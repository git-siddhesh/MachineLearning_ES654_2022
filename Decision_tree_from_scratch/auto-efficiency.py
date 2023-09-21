
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from tree.utils import nested_cross_validation


from metrics import *

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier

np.random.seed(42)

col_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
data = pd.read_fwf('auto-mpg.data', header=None, names=col_names)

data = data[data['horsepower'] != '?'].reset_index()

data['horsepower'] = data['horsepower'].astype('float')


y = data['mpg']
data.drop('car name', axis=1, inplace=True)
X = data.drop('mpg', axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = X[:int(0.8*len(X))], X[int(0.8*len(X)):], y[:int(0.8*len(y))], y[int(0.8*len(y)):]

def get_results(model ,X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    # model.plot()
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)
    if type(y_hat_train) == np.ndarray:
        y_hat_train = pd.Series(y_hat_train)
    y_hat_train.reset_index(drop=True, inplace=True)
    if type(y_hat_test) == np.ndarray:
        y_hat_test = pd.Series(y_hat_test)  
    y_hat_test.reset_index(drop=True, inplace=True)

    y_test.reset_index(drop=True, inplace=True)


    # print('y_hat_test', y_hat_test)
    # print(type(y_hat_test))
    # print('y_test', y_test)
    # print(type(y_test))



    print('Train RMSE: ', rmse(y_hat_train, y_train))
    print('Test RMSE: ', rmse(y_hat_test, y_test))
    print('Train MAE: ', mae(y_hat_train, y_train))
    print('Test MAE: ', mae(y_hat_test, y_test))
    print()

# print('\n DEPTH',5)
def get_scores(trainX, trainY, testX, testY, depth, criteria):
    tree = DecisionTree(criterion=criteria, max_depth=depth) #Split based on Inf. Gain
    tree.fit(trainX, trainY)
    y_hat = tree.predict(testX)

    y_hat.reset_index(drop=True, inplace=True)
    testY.reset_index(drop=True, inplace=True)

    # print('y_hat')
    # print(y_hat)
    # print('testY')
    # print(testY)
    acc = rmse(y_hat, testY)
    return acc

outerFolds, innerFolds, depth_level, depth_step = 5, 4, 18, 2 
res = nested_cross_validation(X,y,None, get_scores, outerFolds, innerFolds, depth_level, depth_step, check_rmse=True)

optimal_depth = max(res, key = lambda x: x[1])[0]
# optimal_depth = 15
model1 = DecisionTree(criterion=None, max_depth=optimal_depth) #Split based on Inf. Gain

print("RESULTS FOR OUR DECISION TREE IMPLEMENTATION")
get_results(model1, X_train, X_test, y_train, y_test)


model2 = DecisionTreeRegressor()
print("RESULTS FOR SKLEARN DECISION TREE IMPLEMENTATION")
get_results(model2, X_train, X_test, y_train, y_test)


# OPTIMAL DEPTH: (15)

# Train RMSE:  10.58127193627844
# Test RMSE:  11.370085189184042
# Train MAE:  8.47
# Test MAE:  9.527906976744186

# Train RMSE:  1.9671692497752686
# Test RMSE:  3.2434445505858878
# Train MAE:  1.4148159594308365
# Test MAE:  2.294136636833298




