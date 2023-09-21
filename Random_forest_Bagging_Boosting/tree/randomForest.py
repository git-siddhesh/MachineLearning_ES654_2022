from numpy.core.fromnumeric import shape

import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd

class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None):

        self.max_depth=max_depth
        self.n_estimators=n_estimators
        self.Forest=[None]*n_estimators

    def fit(self, X, y):

        X_temp1=X.copy()
        X_temp1["res"]=y
        for i in range(self.n_estimators):
            X_temp=X_temp1.sample(frac=0.6)
            Dt=DecisionTreeClassifier(max_features=1)
            Dt.fit(X_temp.iloc[:,:-1],X_temp.iloc[:,-1])
            self.Forest[i]=Dt

    def predict(self, X):

        res=np.zeros((X.shape[0],self.n_estimators))
        for i in range(self.n_estimators):
            Dt=self.Forest[i]
            res[:,i]=np.array(Dt.predict(X))
        y_hat=np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            a=list(res[i])
            y_hat[i]=max(set(a),key=a.count)
        return pd.Series(y_hat)

    def plot(self, criteria):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        for i in range(self.n_estimators):
            tree.plot_tree(self.Forest[i])
            temp="Tree number"+str(i)
            plt.title(temp)
            plt.savefig("./q5/RS_classifier_Tree"+str(i)+criteria+".png")
            # plt.show()


class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators=n_estimators
        self.Forest=[None]*n_estimators

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        X_temp1=X.copy()
        X_temp1["res"]=y
        for i in range(self.n_estimators):
            X_temp=X_temp1.sample(frac=0.6)
            Dt=DecisionTreeRegressor(max_features=1)
            Dt.fit(X_temp.iloc[:,:-1],X_temp.iloc[:,-1])
            self.Forest[i]=Dt

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        res=np.zeros((X.shape[0],self.n_estimators))
        for i in range(self.n_estimators):
            Dt=self.Forest[i]
            res[:,i]=np.array(Dt.predict(X))
        y_hat=np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_hat[i]=np.mean(res[i])
        return pd.Series(y_hat)

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        for i in range(self.n_estimators):
            tree.plot_tree(self.Forest[i])
            temp="Tree number"+str(i)
            plt.title(temp)
            plt.savefig("./q5/RS_Regressor_Tree"+str(i)+".png")
            # plt.show()


# from numpy.core.fromnumeric import shape

# import matplotlib.pyplot as plt
# from sklearn import tree
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import DecisionTreeRegressor
# import numpy as np
# import pandas as pd
 
# class RandomForestClassifier():
#     def __init__(self, n_estimators=100, criterion='gini', max_depth=None, feature_sampling = False):
#         self.n_estimators=n_estimators
#         self.max_depth=max_depth
#         self.criterion=criterion
#         self.Forest_models=[None]*n_estimators
#         self.feature_sampling=feature_sampling
        
#     def fit(self, X, y):
#         """
#         Function to train and construct the RandomForestClassifier
#         Inputs:
#         X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
#         y: pd.Series with rows corresponding to output variable (shape of Y is N)
#         """
#         if self.feature_sampling == False:
#             X1=X.copy()
#             X1["res"]=y
#             # Generate n_estimators number of trees and store them in self.Forest_models
#             for i in range(self.n_estimators):
#                 # Generate a random sample of size 0.7*N from X and y
#                 sampled_input_data=X1.sample(frac=0.7)
#                 # Fit a DecisionTreeClassifier on the sampled data
#                 Dt=DecisionTreeClassifier(max_features=1)
#                 Dt.fit(sampled_input_data.iloc[:,:-1],sampled_input_data.iloc[:,-1])
#                 self.Forest_models[i]=Dt
#         else:
#             n_features = X.shape[1]
#             if self.max_features is None:
#                 self.max_features = n_features
#             for i in range(self.n_estimators):
#                 myDTree = DecisionTreeClassifier(max_depth=self.max_depth, criterion=self.criterion)
#                 # Bagging: selecting the subset of samples:
#                 sample_indices = np.random.choice(n_features, self.max_features, replace=True)
#                 # feature_indices = np.random.choice(n_features, self.max_features, replace=False)
#                 # creating a subset for the features:
#                 X_subset = X[:, sample_indices]
#                 print(X_subset.shape, y.shape)
#                 input()
#                 tree.fit(X_subset, y)
#                 self.trees.append((tree, sample_indices))
#                 self.Forest[i] = tree
#                 self.FeatureSet.append([X_subset, y])
#                 self.models.append(tree)

#     def predict(self, X):
#         """
#         Funtion to run the RandomForestClassifier on a data point
#         Input:
#         X: pd.DataFrame with rows as samples and columns as features
#         Output:
#         y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
#         """
#         res=np.zeros((X.shape[0],self.n_estimators))
#         for i in range(self.n_estimators):
#             Dt=self.Forest_models[i]
#             res[:,i]=np.array(Dt.predict(X))
#         y_hat=np.zeros(X.shape[0])
#         for i in range(X.shape[0]):
#             a=list(res[i])
#             y_hat[i]=max(set(a),key=a.count)
#         return pd.Series(y_hat)

#     def plot(self):
#         """
#         Function to plot for the RandomForestClassifier.
#         It creates three figures

#         1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
#         If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

#         2. Creates a figure showing the decision surface for each estimator

#         3. Creates a figure showing the combined decision surface

#         """
#         for i in range(self.n_estimators):
#             tree.plot_tree(self.Forest_models[i])
#             temp="Tree number"+str(i)
#             plt.title(temp)
#             plt.show()
# '''
#         x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
#         y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
#         xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
#         plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
#         Z = np.array(self.predict(np.c_[xx.ravel(), yy.ravel()]))
#         Z = Z.reshape(xx.shape)
#         cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

#         y_hat=list(self.predict(X))
#         x_axis=list(X.iloc[:,0])
#         y_axis=list(X.iloc[:,1])
#         for i in range(len(x_axis)):
#             if(y_hat[i]==1):
#                 plt.scatter(x_axis[i],y_axis[i],c='RED',cmap=plt.cm.RdYlBu)
#             elif(y_hat[i]==2):
#                 plt.scatter(x_axis[i],y_axis[i],c='BLUE',cmap=plt.cm.RdYlBu)
#             else:
#                 plt.scatter(x_axis[i],y_axis[i],c="GREEN",cmap=plt.cm.RdYlBu)    
#         plt.show()
# '''



# class RandomForestRegressor():
#     def __init__(self, n_estimators=100, criterion='variance', max_depth=None, feature_sampling=False, max_features=None):
#         '''
#         :param n_estimators: The number of trees in the forest.
#         :param criterion: The function to measure the quality of a split.
#         :param max_depth: The maximum depth of the tree.
#         '''
#         self.n_estimators=n_estimators
#         self.Forest_models=[None]*n_estimators
#         self.feature_sampling=feature_sampling
#         self.max_features=max_features
#         self.max_depth=max_depth

#     def fit(self, X, y):
#         """
#         Function to train and construct the RandomForestRegressor
#         Inputs:
#         X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
#         y: pd.Series with rows corresponding to output variable (shape of Y is N)
#         """
        
#         X1=X.copy()
#         X1["res"]=y
#         for i in range(self.n_estimators):
#             sampled_input_data=X1.sample(frac=0.6)
#             Dt=DecisionTreeRegressor(max_features=1)
#             Dt.fit(sampled_input_data.iloc[:,:-1],sampled_input_data.iloc[:,-1])
#             self.Forest_models[i]=Dt

#     def predict(self, X):
#         """
#         Funtion to run the RandomForestRegressor on a data point
#         Input:
#         X: pd.DataFrame with rows as samples and columns as features
#         Output:
#         y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
#         """
#         res=np.zeros((X.shape[0],self.n_estimators))
#         for i in range(self.n_estimators):
#             Dt=self.Forest_models[i]
#             res[:,i]=np.array(Dt.predict(X))
#         y_hat=np.zeros(X.shape[0])
#         for i in range(X.shape[0]):
#             y_hat[i]=np.mean(res[i])
#         return pd.Series(y_hat)

#     def plot(self):
#         """
#         Function to plot for the RandomForestClassifier.
#         It creates three figures

#         1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
#         If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

#         2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

#         3. Creates a figure showing the combined decision surface/prediction

#         """
#         for i in range(self.n_estimators):
#             tree.plot_tree(self.Forest_models[i])
#             temp="Tree number"+str(i)
#             plt.title(temp)
#             plt.show()

