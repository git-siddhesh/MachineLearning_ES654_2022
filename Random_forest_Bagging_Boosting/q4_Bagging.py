# import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from metrics import *
import numpy as np
from ensemble.bagging import BaggingClassifier
from sklearn.model_selection import train_test_split
import time

# Generate a synthetic dataset for classification
X, y = make_classification(n_samples=4000, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#sequencial
classifier_s = BaggingClassifier(base_model=DecisionTreeClassifier, num_estimators=10)

start_time = time.time()
classifier_s.fit(X, y, flag="sequencial")
end_time = time.time()
print("sequencial bagging time", end_time-start_time)


y_hat = classifier_s.predict(X)
print("Accuracy Squencial :", accuracy(y, y_hat, print_sklearn=False))
print("Time Squencial :", round(end_time-start_time, 4), "seconds")
for cls in pd.Series(y).unique():
    print("Precision seqential: ", precision(y_hat, y, cls, print_sklearn=False))
    print("Recall sequential: ", recall(y_hat, y, cls, print_sklearn=False))


classifier_s.plot(X, y, save=True, name="seq")


print("parallel bagging")
#parallel
classifier_p = BaggingClassifier(base_model=DecisionTreeClassifier, num_estimators=10)

start_time = time.time()
classifier_p.fit(X, y, flag="parallel")
end_time = time.time()
print("parallel bagging time", end_time-start_time)

y_hat = classifier_p.predict(X)
print("Accuracy in Parallel :", accuracy(y, y_hat, print_sklearn=False))
print("Time in Parallel :", round(end_time-start_time, 4), "seconds")

for cls in pd.Series(y).unique():
    print("Precision parellel: ", precision(y_hat, y, cls, print_sklearn=False))
    print("Recall parellel: ", recall(y_hat, y, cls, print_sklearn=False))

classifier_p.plot(X, y, save=True, name="par")