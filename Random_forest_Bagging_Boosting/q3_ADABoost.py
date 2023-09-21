import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
from ensemble.ADABoost import AdaBoostClassifier
from sklearn.datasets import make_classification

# np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################

N = 100
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
#X = pd.DataFrame(np.abs(np.random.randn(N, P)))
#y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")
X, y = make_classification(n_samples=N, n_features=P, n_informative=P, n_redundant=0, random_state=11, n_classes=NUM_OP_CLASSES)
# X = np.abs(np.random.randn(N, P))
# y = np.random.randint(NUM_OP_CLASSES, size=N)
print("X.shape: ", X.shape)
print("y.shape: ", y.shape)


Classifier_AB = AdaBoostClassifier(n_estimators=n_estimators)
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
# [fig1, fig2] = Classifier_AB.plot(X, y)
fig11, fig12 = Classifier_AB.plot(X, y)
fig21, fig22 = Classifier_AB.plot2(X, y)
# print("Criteria :", criteria)

y = pd.Series(y)
y_hat = pd.Series(y_hat)
print("Accuracy: ", accuracy(y_hat, y))
# print("Accuracy: ", accuracy_score(y_hat, y))
for cls in y.unique():
    print("Precision: ", precision(y_hat, y, cls))
    print("Recall: ", recall(y_hat, y, cls))

#------------------------------------------------------------
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from metrics import *

# from ensemble.ADABoost import AdaBoostClassifier
# #from tree.base import DecisionTree
# from sklearn.tree import DecisionTreeClassifier
# # Or you could import sklearn DecisionTree
# from sklearn.datasets import load_iris
# np.random.seed(42)

# ########### AdaBoostClassifier on Real Input and Discrete Output ###################




# n_estimators = 3
# iris = load_iris()
# for pair in ([0, 1], [0, 2], [2, 3]):
#     X = pd.DataFrame(iris.data[:,pair])
#     y = pd.Series(iris.target,dtype="category")
#     #print(X)
#     NUM_OP_CLASSES = len(np.unique(y))
#     criteria = "entropy"
#     tree = DecisionTreeClassifier
#     Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators, classes=NUM_OP_CLASSES)
#     Classifier_AB.fit(X, y)
#     y_hat = Classifier_AB.predict(X)
#     #print(y_hat)
#     [fig1, fig2] = Classifier_AB.plot(X,y)
#     print("Criteria :", criteria)
#     print("Accuracy: ", accuracy(y_hat, y))
#     #print(y_hat)
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.metrics import accuracy_score
# for pair in ([0, 1], [0, 2], [2, 3]):
#     X = pd.DataFrame(iris.data[:,pair])
#     y = pd.Series(iris.target)
#     Classifier_AB = AdaBoostClassifier(n_estimators=n_estimators)
#     Classifier_AB.fit(X, y)
#     y_hat = Classifier_AB.predict(X)
#     print(accuracy_score(y,y_hat))
# exit()
# for cls in y.unique():
#     print("Precision: ", precision(y_hat, y, cls))
#     print("Recall: ", recall(y_hat, y, cls))
