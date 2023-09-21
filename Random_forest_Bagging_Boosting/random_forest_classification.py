import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd

class RandomForestClassifierPlot:
    def __init__(self, n_estimators=100, max_depth=None, max_features=None, criterion='gini'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.criterion = criterion
        self.Forest = [None]*n_estimators
        self.FeatureSet = []
        self.models = []
    
    def fit(self, X, y):
        n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = n_features
        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, criterion=self.criterion)
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            X_subset = X[:, feature_indices]
            tree.fit(X_subset, y)
            self.trees.append((tree, feature_indices))
            self.Forest[i] = tree
            self.FeatureSet.append([X_subset, y])
            self.models.append(tree)
    
    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.trees)))
        for i, (tree, feature_indices) in enumerate(self.trees):
            X_subset = X[:, feature_indices]
            predictions[:, i] = tree.predict(X_subset)
        return np.mean(predictions, axis=1)

    def plot(self):
        # Get the number of trees in the random forest
        n_estimators = self.n_estimators

        # Create a figure with subplots for each tree
        fig, axes = plt.subplots(1, n_estimators, figsize=(5 * n_estimators, 5))

        # Loop over each tree in the random forest
        for i in range(n_estimators):
            # Get the ith tree from the random forest
            t = self.Forest[i]

            # Plot the decision surface for the ith tree
            tree.plot_tree(t, ax=axes[i])

            # Set the subplot title
            axes[i].set_title(f"Tree {i+1}")


        # Set the figure title
        fig.suptitle("Decision Surfaces for Random Forest", fontsize=16)

        # Show the plot
        plt.savefig('./q5/RandomForest1.png')
        # plt.show()

        self.plot_surface(self.FeatureSet[i][0], self.FeatureSet[i][1])


    
    def plot_surface(self, X, y):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        # Define the mesh grid for plotting
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Plot the decision surface of each estimator
        fig, axs = plt.subplots(1, self.n_estimators, figsize=(self.n_estimators * 3, 3), sharex=True, sharey=True)
        for i, clf in enumerate(self.models):
            # Create a subplot for the current estimator
            ax = axs[i]

            # Plot the decision surface
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.4)

            # Plot the training data
            ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')

            # Set the title of the subplot
            ax.set_title(f'Estimator {i+1}')

        # Show the plot
        plt.savefig('./q5/RandomForestEstimators1.png')
        # plt.show()

        # Plot the combined decision surface

        # Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        # Z = Z.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, alpha=0.4)
        # plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        # plt.title('Combined decision surface')

        # # Show the plot
        # plt.show()

        fig, axs = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True)
        for i, clf in enumerate(self.models):
            # Create a subplot for the current estimator
            ax = axs

            # Plot the decision surface
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=1/self.n_estimators)

            # Plot the training data
            ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')

            # Set the title of the subplot
        
            ax.set_title(f'Combined Plot')

        # Show the plot
        plt.savefig('./q5/RandomForestCombined1.png')
        # plt.show()

            


class RandomForestRegressorPlot():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators=n_estimators
        self.Forest=[None]*n_estimators
        self.models = []

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
            self.models.append(Dt)
            

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
        # Get the number of trees in the random forest
        n_estimators = self.n_estimators

        # Create a figure with subplots for each tree
        fig, axes = plt.subplots(1, n_estimators, figsize=(5 * n_estimators, 5))

        # Loop over each tree in the random forest
        for i in range(n_estimators):
            # Get the ith tree from the random forest
            t = self.Forest[i]

            # Plot the decision surface for the ith tree
            tree.plot_tree(t, ax=axes[i], filled=True, rounded=True, class_names=True)

            # Set the subplot title
            axes[i].set_title(f"Tree {i+1}")

        # Set the figure title
        fig.suptitle("Decision Surfaces for Random Forest", fontsize=16)

        # Show the plot
        plt.savefig('./q5/RandomForest2.png')
        # plt.show()

        self.plot_surface(self.FeatureSet[i][0], self.FeatureSet[i][1])
        

    def plot_surface(self, X, y):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        # Define the mesh grid for plotting
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Plot the decision surface of each estimator
        fig, axs = plt.subplots(1, self.n_estimators, figsize=(self.n_estimators * 3, 3), sharex=True, sharey=True)
        for i, clf in enumerate(self.models):
            # Create a subplot for the current estimator
            ax = axs[i]

            # Plot the decision surface
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.4)

            # Plot the training data
            ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')

            # Set the title of the subplot
            ax.set_title(f'Estimator {i+1}')

        # Show the plot
        plt.savefig('./q5/RandomForestEstimators1_regressor.png')
        # plt.show()

        # Plot the combined decision surface

        # Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        # Z = Z.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, alpha=0.4)
        # plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        # plt.title('Combined decision surface')

        # # Show the plot
        # plt.show()

        fig, axs = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True)
        for i, clf in enumerate(self.models):
            # Create a subplot for the current estimator
            ax = axs

            # Plot the decision surface
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=1/self.n_estimators)

            # Plot the training data
            ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')

            # Set the title of the subplot
        
            ax.set_title(f'Combined Plot')

        # Show the plot
        plt.savefig('./q5/RandomForestCombined1_regressor.png')
        # plt.show()



# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import DecisionTreeRegressor
# import matplotlib.pyplot as plt
# from sklearn import tree
# import pandas as pd

# class RandomForestClassifierPlot:
#     def __init__(self, n_estimators=100, max_depth=None, max_features=None, criterion='gini'):
#         self.n_estimators = n_estimators
#         self.max_depth = max_depth
#         self.max_features = max_features
#         self.trees = []
#         self.criterion = criterion
#         self.Forest = [None]*n_estimators
#         self.FeatureSet = []
#         self.models = []
    
#     def fit(self, X, y):
#         n_features = X.shape[1]
#         if self.max_features is None:
#             self.max_features = n_features
#         for i in range(self.n_estimators):
#             myDTree = DecisionTreeClassifier(max_depth=self.max_depth, criterion=self.criterion)
#             # Bagging: selecting the subset of samples:
#             sample_indices = np.random.choice(n_features, self.max_features, replace=True)
#             # feature_indices = np.random.choice(n_features, self.max_features, replace=False)
#             # creating a subset for the features:
#             X_subset = X[:, sample_indices]
#             print(X_subset.shape, y.shape)
#             input()
#             tree.fit(X_subset, y)
#             self.trees.append((tree, sample_indices))
#             self.Forest[i] = tree
#             self.FeatureSet.append([X_subset, y])
#             self.models.append(tree)
    
#     def predict(self, X):
#         predictions = np.zeros((X.shape[0], len(self.trees)))
#         for i, (tree, feature_indices) in enumerate(self.trees):
#             X_subset = X[:, feature_indices]
#             predictions[:, i] = tree.predict(X_subset)
#         return np.mean(predictions, axis=1)

#     def plot(self):
#         # Get the number of trees in the random forest
#         n_estimators = self.n_estimators

#         # Create a figure with subplots for each tree
#         fig, axes = plt.subplots(1, n_estimators, figsize=(5 * n_estimators, 5))

#         # Loop over each tree in the random forest
#         for i in range(n_estimators):
#             # Get the ith tree from the random forest
#             t = self.Forest[i]

#             # Plot the decision surface for the ith tree
#             tree.plot_tree(t, ax=axes[i])

#             # Set the subplot title
#             axes[i].set_title(f"Tree {i+1}")


#         # Set the figure title
#         fig.suptitle("Decision Surfaces for Random Forest", fontsize=16)

#         # Show the plot
#         plt.show()

#         self.plot_surface(self.FeatureSet[i][0], self.FeatureSet[i][1])


    
#     def plot_surface(self, X, y):
#         """
#         Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
#         Creates two figures
#         Figure 1 consists of 1 row and `n_estimators` columns
#         The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
#         Further, the scatter plot should have the marker size corresponnding to the weight of each point.

#         Figure 2 should also create a decision surface by combining the individual estimators

#         Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

#         This function should return [fig1, fig2]
#         """
#         # Define the mesh grid for plotting
#         h = 0.02
#         x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#         y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#         xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#         # Plot the decision surface of each estimator
#         fig, axs = plt.subplots(1, self.n_estimators, figsize=(self.n_estimators * 3, 3), sharex=True, sharey=True)
#         for i, clf in enumerate(self.models):
#             # Create a subplot for the current estimator
#             ax = axs[i]

#             # Plot the decision surface
#             Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#             Z = Z.reshape(xx.shape)
#             ax.contourf(xx, yy, Z, alpha=0.4)

#             # Plot the training data
#             ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')

#             # Set the title of the subplot
#             ax.set_title(f'Estimator {i+1}')

#         # Show the plot
#         plt.show()

#         # Plot the combined decision surface

#         # Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
#         # Z = Z.reshape(xx.shape)
#         # plt.contourf(xx, yy, Z, alpha=0.4)
#         # plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
#         # plt.title('Combined decision surface')

#         # # Show the plot
#         # plt.show()

#         fig, axs = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True)
#         for i, clf in enumerate(self.models):
#             # Create a subplot for the current estimator
#             ax = axs

#             # Plot the decision surface
#             Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#             Z = Z.reshape(xx.shape)
#             ax.contourf(xx, yy, Z, alpha=1/self.n_estimators)

#             # Plot the training data
#             ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')

#             # Set the title of the subplot
        
#             ax.set_title(f'Combined Plot')

#         # Show the plot
#         plt.show()

            


# class RandomForestRegressorPlot():
#     def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
#         '''
#         :param n_estimators: The number of trees in the forest.
#         :param criterion: The function to measure the quality of a split.
#         :param max_depth: The maximum depth of the tree.
#         '''
#         self.n_estimators=n_estimators
#         self.Forest=[None]*n_estimators

#     def fit(self, X, y):
#         """
#         Function to train and construct the RandomForestRegressor
#         Inputs:
#         X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
#         y: pd.Series with rows corresponding to output variable (shape of Y is N)
#         """
#         X_temp1=X.copy()
#         X_temp1["res"]=y
#         for i in range(self.n_estimators):
#             X_temp=X_temp1.sample(frac=0.6)
#             Dt=DecisionTreeRegressor(max_features=1)
#             Dt.fit(X_temp.iloc[:,:-1],X_temp.iloc[:,-1])
#             self.Forest[i]=Dt

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
#             Dt=self.Forest[i]
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
#         # Get the number of trees in the random forest
#         n_estimators = self.n_estimators

#         # Create a figure with subplots for each tree
#         fig, axes = plt.subplots(1, n_estimators, figsize=(5 * n_estimators, 5))

#         # Loop over each tree in the random forest
#         for i in range(n_estimators):
#             # Get the ith tree from the random forest
#             t = self.Forest[i]

#             # Plot the decision surface for the ith tree
#             tree.plot_tree(t, ax=axes[i], filled=True, rounded=True, class_names=True)

#             # Set the subplot title
#             axes[i].set_title(f"Tree {i+1}")

#         # Set the figure title
#         fig.suptitle("Decision Surfaces for Random Forest", fontsize=16)

#         # Show the plot
#         plt.show()


#     def plot_surface(self, X, y):
#         """
#         Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
#         Creates two figures
#         Figure 1 consists of 1 row and `n_estimators` columns
#         The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
#         Further, the scatter plot should have the marker size corresponnding to the weight of each point.

#         Figure 2 should also create a decision surface by combining the individual estimators

#         Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

#         This function should return [fig1, fig2]
#         """
#         # Define the mesh grid for plotting
#         h = 0.02
#         x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#         y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#         xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#         # Plot the decision surface of each estimator
#         fig, axs = plt.subplots(1, self.n_estimators, figsize=(self.n_estimators * 3, 3), sharex=True, sharey=True)
#         for i, clf in enumerate(self.models):
#             # Create a subplot for the current estimator
#             ax = axs[i]

#             # Plot the decision surface
#             Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#             Z = Z.reshape(xx.shape)
#             ax.contourf(xx, yy, Z, alpha=0.4)

#             # Plot the training data
#             ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')

#             # Set the title of the subplot
#             ax.set_title(f'Estimator {i+1}')

#         # Show the plot
#         plt.show()

#         # Plot the combined decision surface

#         # Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
#         # Z = Z.reshape(xx.shape)
#         # plt.contourf(xx, yy, Z, alpha=0.4)
#         # plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
#         # plt.title('Combined decision surface')

#         # # Show the plot
#         # plt.show()

#         fig, axs = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True)
#         for i, clf in enumerate(self.models):
#             # Create a subplot for the current estimator
#             ax = axs

#             # Plot the decision surface
#             Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#             Z = Z.reshape(xx.shape)
#             ax.contourf(xx, yy, Z, alpha=1/self.n_estimators)

#             # Plot the training data
#             ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')

#             # Set the title of the subplot
        
#             ax.set_title(f'Combined Plot')

#         # Show the plot
#         plt.show()

