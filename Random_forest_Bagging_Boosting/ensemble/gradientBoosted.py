import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

class GradientBoostedRegressor:
    def __init__(
        self, base_estimator, n_estimators=3, learning_rate=0.1, max_depth=2
    ):  # Optional Arguments: Type of estimator
        """
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        :param learning_rate: The learning rate shrinks the contribution of each tree by `learning_rate`.
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.alphas = []
        self.avg_pred = 0
        self.max_depth = max_depth

        

    def fit(self, X, y):
        """
        Function to train and construct the GradientBoostedRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        # n_samples = X.shape[0]
        y_residual = y.copy()
        self.avg_pred = np.mean(y)
        # print("AVG",self.avg_pred)
        y_residual = y_residual - self.avg_pred
        # print("Residuals After avg:\n", y_residual)
        for _ in range(self.n_estimators):
            # for now we are assuming that the base estimator is a decision tree
            model = self.base_estimator(max_depth=self.max_depth)
            # model = DecisionTreeRegressor(max_depth=3)
            model.fit(X, y_residual)
            # tree.plot_tree(model)
            y_pred = model.predict(X)
            # print("Y_pred: \n", y_pred)
            y_residual = y_residual - self.learning_rate*(y_pred)
            # print("Residuals: \n", y_residual)
            self.models.append(model)
            self.alphas.append(self.learning_rate)

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        model_preds = np.array([model.predict(X) for model in self.models])
        # print("Model preds: \n", model_preds)
        # print("Alphas: \n", self.alphas)
        # print("Avg pred: \n", self.avg_pred)
        # print(np.dot(self.alphas, model_preds))

        return self.avg_pred + np.dot(self.alphas, model_preds)
    
    def plot(self, X, y):
        y_pred = self.predict(X)
        plt.plot(X, y, 'o')
        plt.plot(X, y_pred, 'X')
        plt.savefig("./q6/gradientBoosted.png")
        plt.show()
