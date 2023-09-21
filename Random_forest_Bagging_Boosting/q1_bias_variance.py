import numpy as np

from tree.base import DecisionTree
import matplotlib.pyplot as plt
from metrics import *
from tree.utils import nested_cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import bias_variance_decomp


np.random.seed(1234)
x = np.linspace(0, 10, 50)
eps = np.random.normal(0, 5, 50)

y = x**2 + 1 + eps
y = pd.Series(y)
X = pd.DataFrame(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def calculate_bias(y,y_hat):
    # return np.mean((y - np.mean(y_hat))**2)
    return np.mean((y - y_hat)**2)

def calculate_variance(y_hat):
    variance = np.mean((np.mean(y_hat) - y_hat)**2)
    # variance = np.var(y_hat)
    return variance

def normalize(x):
    return (x - x.min())/(x.max() - x.min())

depth = np.arange(1, 15)
train_mses = []
test_mses = []
biases_on_complete_data = []
variances_on_complete_data = []
mses = []

for d in depth:
    dt = DecisionTreeRegressor(max_depth=d)
    dt.fit(X_train, y_train)

    y_pred_train = dt.predict(X_train)
    y_pred_test = dt.predict(X_test)

    y_pred = dt.predict(X)


    bias = calculate_bias(y_test, y_pred_test)
    variance = calculate_variance(y_pred_test)

    print(f"Depth = {d}, Bias = {bias}, Variance = {variance}")
    # bias = calculate_bias(y, y_pred)
    # variance = calculate_variance(y_pred)
    # bias = sse - variance
    biases_on_complete_data.append(bias)
    variances_on_complete_data.append(variance)

    train_mse = mse(y_train, y_pred_train)
    test_mse = mse(y_test, y_pred_test)

    print(f"Depth = {d}, Train MSE = {train_mse}, Test MSE = {test_mse}")

    train_mses.append(train_mse)
    test_mses.append(test_mse)

    mse_x = mse(y_pred, y)
    mses.append(mse_x)

    # print(test_mse, variance, mse_x)

biases_on_complete_data = np.array(biases_on_complete_data)
variances_on_complete_data = np.array(variances_on_complete_data)
normalized_biases = normalize(biases_on_complete_data)
# print(normalized_biases)
normalized_variances = normalize(variances_on_complete_data)
# biases_on_complete_data = (biases_on_complete_data - biases_on_complete_data.min())/(biases_on_complete_data.max() - biases_on_complete_data.min())
# variances_on_complete_data = (variances_on_complete_data - variances_on_complete_data.min())/(variances_on_complete_data.max() - variances_on_complete_data.min())

# train_mses = np.array(train_mses)
# test_mses = np.array(test_mses)
# train_mses = (train_mses - train_mses.min())/(train_mses.max() - train_mses.min())
# test_mses = (test_mses - test_mses.min())/(test_mses.max() - test_mses.min())

plt.plot(depth, train_mses, label='Train error')
plt.plot(depth, test_mses, label='Test error')
plt.title('Bias-Variance Tradeoff with Decision Tree Depth')
plt.xlabel('Tree Depth')
plt.ylabel('MSE')
plt.legend()
plt.savefig('./q1/q1_train_test_error.png')
plt.show()



# Plot the bias-variance tradeoff curve
plt.plot(depth, normalized_biases, label='Bias')
plt.plot(depth, normalized_variances, label='Variance')
# plt.plot(depth, train_mses, label='Train error')
# plt.plot(depth, test_mses, label='Test error')
plt.title('Bias-Variance Tradeoff with Decision Tree Depth')
plt.xlabel('Tree Depth')
plt.ylabel('MSE')
plt.legend()
plt.savefig('./q1/q1_bias_variance.png')
plt.show()
