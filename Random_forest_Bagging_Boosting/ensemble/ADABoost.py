
import numpy as np
from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import export_tree
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap


class AdaBoostClassifier():
    def __init__(self, n_estimators=3, base_estimator = DecisionTreeClassifier, classes = 2): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.classes = classes
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        # Calculate the number of samples
        n_samples = X.shape[0]

        # Initialize the weights to 1/N for each samples
        sample_weights = np.full(n_samples, (1 / n_samples))

        for _ in range(self.n_estimators):
            my_model = self.base_estimator(max_depth=1)
            # my_model = DecisionTreeClassifier(max_depth=1)
            my_model.fit(X, y, sample_weight=sample_weights)
            y_pred = my_model.predict(X)
            # Calculate the error
            error = np.sum(sample_weights[y_pred != y])/ np.sum(sample_weights)
            # Calculate the alpha
            alpha = 0.5 * np.log((1 - error) / error)
            # Update the weights
            # for samples which are predicted correctly, the weight is reduced
            # for samples which are predicted incorrectly, the weight is increased
            sample_weights[y_pred != y] *= np.exp(alpha)
            sample_weights[y_pred == y] *= np.exp(-alpha)
            # Normalize the weights
            sample_weights /= np.sum(sample_weights)
            # Add the model and alpha to the list
            self.models.append(my_model)
            self.alphas.append(alpha)


    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        # create the prediction of each class for each model
        # model_preds = np.array([model.predict(X) for model in self.models])
        # create the weighted sum of the predictions
        # weighted_preds = np.dot(self.alphas, model_preds)

        weight_class = np.zeros((X.shape[0], self.classes))
        for i, model in enumerate(self.models):
            y_hat = pd.Series(model.predict(X))
            weight_class[range(y_hat.size), y_hat] += self.alphas[i]
        
        prediction = np.argmax(weight_class, axis=1)
        return pd.Series(prediction)
        # prediction = []
        # weight_class=[{j:0 for j in range(self.classes)} for i in range(X.shape[0])]
        # for i in range(len(self.models)):
        #     y_hat = pd.Series(self.models[i].predict(X))
        #     for j in range(y_hat.size):
        #         weight_class[j][y_hat.iat[j]]+=self.weights[i]

        
        # #print(weight_class)
        # for i in weight_class:
        #     prediction.append(max(i, key = i.get)  )

        # write above code in one line



        # return the prediction based on max weighted output class


        # model_preds = np.array([model.predict(X) for model in self.models])
        # return pd.Series(np.sign(np.dot(self.alphas, model_preds)))

    def plot(self, X, y):
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
        fig1, axs = plt.subplots(1, self.n_estimators, figsize=(self.n_estimators * 3, 3), sharex=True, sharey=True)
        for i, clf in enumerate(self.models):
            # Create a subplot for the current estimator
            ax = axs[i]

            # Plot the decision surface
            Z = np.array(clf.predict(np.c_[xx.ravel(), yy.ravel()]))
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.4)

            # Plot the training data
            ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')

            # Set the title of the subplot
            ax.set_title(f'Estimator {i+1}')

        # Show the plot
        plt.savefig("./q3/individual_decision_surfaces.png")
        plt.show()

        # Plot the combined decision surface

        Z = np.array(self.predict(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        plt.title('Combined decision surface')

        # Show the plot
        plt.savefig("./q3/combined_decision_surface.png")
        plt.show()
        return fig1, plt

    def plot2(self, X, y):
        color = ["r", "b", "g","y"]
        plot_step = 0.02
        plot_step_coarser = 0.5
        cmap = plt.cm.RdYlBu
        Zs = None
        fig1, ax1 = plt.subplots(
            1, len(self.models), figsize=(5*len(self.models), 4))
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
        X = pd.DataFrame(X)
        y = pd.Series(y, dtype='category')
        x_min, x_max = X.iloc[:, 0].min()-1, X.iloc[:, 0].max()+1
        y_min, y_max = X.iloc[:, 1].min()-1, X.iloc[:, 1].max()+1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                                 np.arange(y_min, y_max, plot_step))

        
        self.alpha = (self.alphas-min(self.alphas))/(max(self.alphas)-min(self.alphas))
        for i, (alpha_m, tree) in enumerate(zip(self.alpha, self.models)):
            print("-----------------------------")
            print("Tree Number: {}".format(i+1))
            print("-----------------------------")
            print(sklearn.tree.export_text(tree))
            
            # _ = ax1.add_subplot(1, len(self.models), i + 1)
            ax1[i].set_ylabel("X2")
            ax1[i].set_xlabel("X1")
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            if Zs is None:
                Zs = alpha_m*Z
            else:
                Zs += alpha_m*Z
            cs = ax1[i].contourf(xx, yy, Z, cmap=cmap)
            cs1 = ax2.contourf(xx, yy, Z, alpha = alpha_m, cmap=cmap)
            xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min, x_max, plot_step_coarser),np.arange(y_min, y_max, plot_step_coarser),)
            Z_points_coarser = tree.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape)
            ax1[i].scatter(xx_coarser,yy_coarser,s=30,c=Z_points_coarser,cmap=cmap,edgecolors="black",)
            fig1.colorbar(cs, ax=ax1[i], shrink=0.9)
            ax2.scatter(xx_coarser,yy_coarser,s=30,c=Z_points_coarser,cmap=cmap,edgecolors="black",)
        
            for y_label in range(self.classes):
                idx = y == y_label
                id = list(y.cat.categories).index(y[idx].iloc[0])

                ax1[i].scatter(X[idx].iloc[:, 0], X[idx].iloc[:, 1], c=color[id],cmap=ListedColormap(["r", "g", "b"]), edgecolor='black', s=20,label="Class: "+str(y_label))
            ax1[i].set_title("Decision Surface Tree: " + str(i+1))
            ax1[i].legend()
        fig1.tight_layout()

        # For Common surface
        '''fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
        com_surface = Z
        cs = ax2.contourf(xx, yy, com_surface, cmap=cmap)
        xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min, x_max, plot_step_coarser),np.arange(y_min, y_max, plot_step_coarser),)
        Z_points_coarser = tree.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape)
        '''#ax2.scatter(xx_coarser,yy_coarser,s=30,c=Z_points_coarser,cmap=cmap,edgecolors="black",)
        for y_label in range(self.classes):
            idx = y == y_label
            id = list(y.cat.categories).index(y[idx].iloc[0])
            # print(color[id])
            # input()
            ax2.scatter(X[idx].iloc[:, 0], X[idx].iloc[:, 1], c=y[idx],cmap=cmap, edgecolor='black', s=30,label="Class: "+str(y_label))
        
        ax2.set_ylabel("X2")
        ax2.set_xlabel("X1")
        ax2.legend(loc="lower right")
        ax2.set_title("Common Decision Surface")
        fig2.colorbar(cs1, ax=ax2, shrink=0.9)


        # Saving Figures
        fig1.savefig(fname='./q3/fig1.png')
        fig2.savefig(fname='./q3/fig2.png')
        return fig1, fig2
