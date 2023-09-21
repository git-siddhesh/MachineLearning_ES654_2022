import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from tree.utils import nested_cross_validation

np.random.seed(42)

# Read dataset
# ...
# 

from sklearn.datasets import make_classification
X, y = make_classification(
n_features=2, n_redundant=0 ,n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)



X_train, X_test, y_train, y_test = pd.DataFrame(X[:int(0.7*len(X))]), pd.DataFrame(X[int(0.7*len(X)):]), pd.Series(y[:int(0.7*len(y))], dtype = 'category'), pd.Series(y[int(0.7*len(y)):], dtype = 'category')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# For plotting


criteria = 'information_gain'
tree = DecisionTree(criterion=criteria, max_depth=20) #Split based on Inf. Gain
tree.fit(X_train, y_train)
tree.plot()

y_hat_train = tree.predict(X_train)
print('Training Results:')
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat_train, y_train))
for cls in y_train.unique():
    print('Class: ', cls)
    print('Precision: ', precision(y_hat_train, y_train, cls))
    print('Recall: ', recall(y_hat_train, y_train, cls))
    print()

y_hat = tree.predict(X_test)
print('Testing Results:')
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y_test.unique():
    print('Class: ', cls)
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))
    print()







# arr_x = [i+1 for i in range(len(y))]
plt.scatter(X[:,0], X[:,1], c = y)
# plt.scatter(X[:,0], y, label='y')
# plt.scatter(X[:int(0.7*len(y)), 0], y_hat_train, label='y_train_pred')
# plt.scatter(X[int(0.7*len(y)):, 0], y_hat, label='y_test_pred')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Classification Data')
plt.legend()
plt.show()


# ###########################################################

X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

def get_scores(trainX, trainY, testX, testY, depth, criteria):
    tree = DecisionTree(criterion=criteria, max_depth=depth) #Split based on Inf. Gain
    # tree.fit(X_train, y_train)
    # y_hat = tree.predict(X_test)
    # acc = accuracy(y_hat, y_test)
    tree.fit(trainX, trainY)
    y_hat = tree.predict(testX)
    # print(y_hat)
    # print(testY)
    acc = accuracy(y_hat.reset_index(drop=True), testY.reset_index(drop=True))
    return acc

criteria = 'information_gain'
# criteria = 'gini_index' 

'''
The nested cross-validation function defination is written in tree/utils.py
'''
# def nested_cross_validation(X, y, criteria, get_scores_func):

#     foldX = [X[int(i/5 *len(X)) : int((i+1)/5 *len(X))] for i in range(5)]
#     foldy = [y[int(i/5 *len(y)) : int((i+1)/5 *len(y))] for i in range(5)]
#     testing_score = []
#     for outerFold in range(1, 6):
#         testX = pd.DataFrame(foldX[outerFold-1])
#         testY = pd.Series(foldy[outerFold-1])
#         trainX = pd.concat([pd.DataFrame(foldX[(outerFold+i)%5]) for i in range(1,5)]).reset_index(drop=True)
#         trainY = pd.concat([pd.Series(foldy[(outerFold+1)%5]) for i in range(1,5) ]).reset_index(drop=True)

#         foldX_inner = [trainX[int(i/5 *len(X)) : int((i+1)/5 *len(X))] for i in range(5)]
#         foldy_inner = [trainY[int(i/5 *len(y)) : int((i+1)/5 *len(y))] for i in range(5)]

#         depth_accuracy_map = dict()
#         for depth in range(0,6):
#             validation_score = []
#             for innerfold in range(1, 6):
#                 valdX = pd.DataFrame(foldX_inner[innerfold-1])
#                 valdY = pd.Series(foldy_inner[innerfold-1])
#                 trainX_inner = pd.concat([pd.DataFrame(foldX_inner[(outerFold+i)%5]) for i in range(1,5)]).reset_index(drop=True)
#                 trainY_inner = pd.concat([pd.Series(foldy_inner[(outerFold+1)%5]) for i in range(1,5) ]).reset_index(drop=True)
                
#                 validation_score.append(get_scores_func(trainX_inner, trainY_inner, valdX, valdY, depth, criteria))

#             depth_accuracy_map[depth] = np.mean(validation_score)
#         print(depth_accuracy_map)
#         max_depth_accuracy = max(depth_accuracy_map, key= lambda x: depth_accuracy_map[x])

#         testing_score.append((max_depth_accuracy, get_scores_func(trainX, trainY, testX, testY, max_depth_accuracy, criteria)))
#     return testing_score

testing_score = nested_cross_validation(X, y, criteria, get_scores, 5, 4, 10, 1)

print("TRAINING SCORE: ",testing_score)