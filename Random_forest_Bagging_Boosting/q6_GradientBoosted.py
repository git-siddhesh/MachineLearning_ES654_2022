import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
# from ensemble.gradientBoosted import GradientBoostedClassifier
from ensemble.gradientBoosted import GradientBoostedRegressor
from tree.base import DecisionTree
from sklearn.model_selection import train_test_split

# Or use sklearn decision tree

########### GradientBoostedClassifier ###################

from sklearn.datasets import make_regression

X1, y1= make_regression(
       n_features=3,
       n_informative=3,
       noise=10,
       tail_strength=10,
       random_state=42,
   )

# For plotting
import matplotlib.pyplot as plt
plt.scatter(X1[:, 0], y1)

plt.show()
# N = 5
# P = 5
# X1 = np.random.randint(P, size = (N,3))
# y1 = np.random.randn(N)


# height = np.array([1.6, 1.6, 1.5, 1.8, 1.5, 1.4])
# color = np.array([1, 2, 1, 3, 2, 1])
# gender = np.array([1, 0, 0, 1, 1, 0])

# X1 = pd.DataFrame({'height': height, 'color':color, 'gender': gender}) 
# y1 = pd.Series([88, 76,56, 73, 77, 57], dtype=np.float64)
min_mse = 100000
max_r2 = 0
optimal_params_on_mse = []
optimal_params_on_r2 = []
for lr in [0.001, 0.01, 0.1, 1]:
    for estimators in [1, 10, 100]:
        for depth in [1, 2, 3]:
            GBreg = GradientBoostedRegressor(base_estimator=DecisionTreeRegressor, n_estimators=estimators, learning_rate=lr, max_depth=depth)
            GBreg.fit(X1, y1)
            y_hat = GBreg.predict(X1)
            # print("Prediction: ", y_hat)
            # print("Actual: ", y1)
            # print(y1-y_hat)
            mserr = mean_squared_error(y1, y_hat)
            r2 = r2_score(y1, y_hat)
            if mserr < min_mse:
                min_mse = mserr
                optimal_params_on_mse = [lr, estimators, depth]
            if r2 > max_r2:
                max_r2 = r2
                optimal_params_on_r2 = [lr, estimators, depth] 
            print("----------------------")
            print("Learning Rate: ", lr, "Estimators: ", estimators, "Depth: ", depth)
            print("MSE: ", mserr)
            print("R2: ", r2)
# GBreg = GradientBoostedRegressor(base_estimator=DecisionTreeRegressor, n_estimators=10, learning_rate=0.1, max_depth=2)
# GBreg.fit(X1, y1)
# y_hat = GBreg.predict(X1)
# # print("Prediction: ", y_hat)
# # print("Actual: ", y1)
# # print(y1-y_hat)
# print("MSE: ", mean_squared_error(y1, y_hat))
# print("R2: ", r2_score(y1, y_hat))

print("Optimal params on MSE: ", optimal_params_on_mse)
print("Optimal params on R2: ", optimal_params_on_r2)
print("Min MSE: ", min_mse)
print("Max R2: ", max_r2)

dreg = DecisionTreeRegressor(max_depth=3)
dreg.fit(X1, y1)
y_hat = dreg.predict(X1)
# print(y1-y_hat)
print("DTR MSE: ", mean_squared_error(y1, y_hat))
print("DTR R2: ", r2_score(y1, y_hat))
############################################


print("Creating a new dataset of single feature to test the plotting function")
X, y= make_regression(
    n_features=1,
    n_informative=1,
    noise=10,
    tail_strength=10,
    random_state=42,
)





plt.scatter(X, y)
plt.savefig('./q6/q6.png')
plt.show()


# # plot the actual and predicted values on 
# hyperparams: lr = 0.1, estimators = 100, depth = 2
reg = GradientBoostedRegressor(base_estimator=DecisionTreeRegressor, n_estimators=optimal_params_on_mse[1], max_depth=optimal_params_on_mse[2], learning_rate=optimal_params_on_mse[0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg.fit(X_train, y_train)
reg.plot(X_test, y_test)


# plt.scatter(y1, label='Actual')
# plt.scatter(y_hat, label='Predicted')
# plt.legend()
# plt.show()
