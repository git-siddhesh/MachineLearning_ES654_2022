# -*- coding: utf-8 -*-
"""linear_regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dSwJGtzoxAFZlW8itoyHW0NtJSze1Pk7
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
from matplotlib import cm

import jax
import jax.numpy as jnp

np.random.seed(45)

class LinearRegression():
  def __init__(self, fit_intercept=True):
    # Initialize relevant variables
    '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
    '''
    self.fit_intercept = fit_intercept 
    self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
    self.all_coef=pd.DataFrame([]) # Stores the thetas for every iteration (theta vectors appended) (for the iterative methods)
    self.all_coef = []


  def modify_data(self, X):
    # Modify the data to include the intercept term
    if self.fit_intercept == True:
      X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    return X

  def fit_sklearn_LR(self, X, y, sample_weights=None, normalize=False):
    # Solve the linear regression problem by calling Linear Regression
    # from sklearn, with the relevant parameters
    # create an array of dimension (n_features, 1) for the coeff_
    # if y.ndim == 1:
    #   self.coef_ = np.zeros((X.shape[1], 1))
    # else:  # if y has multiple targets 
    #   self.coef_ = np.zeros((y.shape[1], X.shape[1]))
    print("X: ", X.shape, "y: ", y.shape)
    reg = LR(fit_intercept=self.fit_intercept, copy_X=True)
    reg.fit(X, y, sample_weight=sample_weights)
    self.coef_ = reg.coef_
    if self.fit_intercept == True:  
      self.coef_ = np.append(reg.intercept_, self.coef_)
    print("COEFFIECIENT SKLEARN: ",self.coef_)
    
  def fit_normal_equations(self, X, y):
    # Solve the linear regression problem using the closed form solution
    # to the normal equation for minimizing ||Wx - y||_2^2

    X = self.modify_data(X)
    self.coef_ = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    print("X: ", X.shape, "y: ", y.shape)

    print("Normal Equation: ",self.coef_)

  def fit_SVD(self, X, y):
    # Solve the linear regression problem using the SVD of the 
    # coefficient matrix

    X = self.modify_data(X)
    U, S, V = np.linalg.svd(X, full_matrices=False)
    self.coef_ = V.T.dot(np.linalg.inv(np.diag(S))).dot(U.T).dot(y) 

    print("X: ", X.shape, "y: ", y.shape)

    print("Normal Equation: ",self.coef_)

  def mse_loss(self, X, y):                
    # Compute the MSE loss with the learned model
    return np.mean((self.predict(X) - y)**2)
  
  def compute_gradient(self, X, y, penalty, alpha=0.1):

    regression_value = 0
    if penalty == 'l2':
      regression_value = 2 * alpha * self.coef_
    n = X.shape[0]
    # print(X.shape, self.coef_.shape, y.shape)
    return (2/n) * X.T.dot(X.dot(self.coef_) - y) + regression_value

  def compute_jax_gradient(self, X, y, penalty, alpha=0.1):

    X = np.array(X)
    y = np.array(y)
    # print(type(X), X.shape)
    # print(type(y), y.shape)

    # input()
    def mse_loss_jax(theta):
      return jnp.mean((jnp.dot(X.astype('float'), theta) - y)**2)
    

  
    if penalty == 'l1':
      return jax.grad(mse_loss_jax)(self.coef_) + alpha * jnp.sign(self.coef_)
    elif penalty == 'l2':
      return jax.grad(mse_loss_jax)(self.coef_) + 2 * alpha * self.coef_
    else:
      return jax.grad(mse_loss_jax)(self.coef_)    # lr : Default learning rate    # implement batch gradient descent

    

  def fit_gradient_descent(self, X, y, batch_size, gradient_type, penalty_type, alpha = 0.1 , num_iters=20, lr=0.01, print_data = True):

    X = self.modify_data(X)

    self.coef_ = np.zeros(X.shape[1])
    # self.all_coef = pd.DataFrame([])
    # append the initial theta vector using pandas concat
    # self.all_coef['0'] = pd.Series(self.coef_)



    # self.all_coef = self.all_coef.append(pd.Series(self.coef_), ignore_index=True)
    
    # self.all_coef.concat(pd.Series(self.coef_), ignore_index=True)
    # self.all_coef = self.all_coef.append(pd.Series(self.coef_), ignore_index=True)

    # implement stochastic/mini-batch gradient descent
    for i in range(num_iters):
      # loop over the batched and batched may not be of divisible size
      for j in range(0, X.shape[0], batch_size):
        if j+batch_size >= X.shape[0]:
          end = X.shape[0]
        else:
          end = j+batch_size
        if gradient_type == 'manual':
          grad = self.compute_gradient(X[j:end], y[j:end], penalty_type, alpha)
        else:
          grad = self.compute_jax_gradient(X[j:end], y[j:end], penalty_type, alpha)
        self.coef_ = self.coef_ - lr * grad
        self.all_coef.append(self.coef_)

      # self.all_coef['i+1'] = pd.Series(self.coef_)

      # self.coef_ to the dataframe self.all_coef
      # self.all_coef = self.all_coef.append(pd.Series(self.coef_), ignore_index=True)
      # self.all_coef.concat(pd.Series(self.coef_), ignore_index=True)
      # self.all_coef = self.all_coef.append(pd.Series(self.coef_), ignore_index=True)
    if print_data:
      print("X: ", X.shape, "y: ", y.shape)

      print("Coeffiecient Gradient Descent ",self.coef_)
    return self.coef_



  def fit_SGD_with_momentum(self, X, y, num_iters=20,  penalty='l2', alpha=0,  beta=0.9, lr=0.01):
    # Solve the linear regression problem using sklearn's implementation of SGD
    # penalty: refers to the type of regularization used (ridge)
    # beta: momentum parameter
    # https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/
    X = self.modify_data(X)
    batch_size = 1 # since we are using SGD
    self.coef_ = np.zeros(X.shape[1])
    self.velocity = np.zeros(X.shape[1])
    for _ in range(num_iters):
      for j in range(0, X.shape[0], batch_size):
        if j+batch_size >= X.shape[0]:
          end = X.shape[0]
        else:
          end = j+batch_size
        grad = self.compute_gradient(X[j:end], y[j:end], penalty, alpha)
        # https://d2l.ai/chapter_optimization/momentum.html#the-momentum-method
        '''Method 1'''

        # self.velocity = beta * self.velocity + lr * grad
        # self.coef_ = self.coef_ - self.velocity
        '''Method 2'''
        # self.velocity = beta * self.velocity + (1 - beta) * grad
        # self.coef_ = self.coef_ - lr * self.velocity
        '''Method 3'''
        self.velocity = beta * self.velocity + grad
        self.coef_ = self.coef_ - lr * self.velocity

    print("X: ", X.shape, "y: ", y.shape)

    print("Coefficients: SGD with momentum ", self.coef_)
  

  def predict(self, X):
    # Funtion to run the LinearRegression on a test data point
    # X = pd.DataFrame(X).copy(deep=True)
    # if self.fit_intercept == True:
    #   X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    X = self.modify_data(X)

    return X.dot(self.coef_)


  # def plot_surface(self, X, y, t_0, t_1):
  #   '''
  #   Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
  #   theta_0 and theta_1 over a range. Indicates the RSS based on given value of theta_0 and theta_1 by a
  #   red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.
  #     :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
  #     :param theta_0: Value of theta_0 for which to indicate RSS #pd Series of all theta_0
  #     :param theta_1: Value of theta_1 for which to indicate RSS #pd Series of all theta_1
  #     :return matplotlib figure plotting RSS
  #   '''
  #   fig = plt.figure()
  #   ax = fig.add_subplot(111, projection='3d')
  #   residual_list, theta_0_list, theta_1_list = [], [], []
    
  #   theta_0_vals = np.linspace(self.coef_[0]-10, self.coef_[0]+10, 100)
  #   theta_1_vals = np.linspace(self.coef_[1]-4, self.coef_[1]+4, 100)
  #   theta_0_grid, theta_1_grid = np.meshgrid(theta_0_vals, theta_1_vals)

  #   rss_vals = np.zeros((theta_0_vals.shape[0], theta_1_vals.shape[0]))
  #   for i, theta_0 in enumerate(theta_0_vals):
  #     for j, theta_1 in enumerate(theta_1_vals):
  #       y_pred = theta_0 + theta_1 * X
  #       residual = y - y_pred
  #       rss = np.sum(residual**2)
  #       rss_vals[i, j] = rss
  #       residual_list.append(rss)
  #       theta_0_list.append(theta_0)
  #       theta_1_list.append(theta_1)
  #       residual_list.append(np.sum((y - (theta_0 + theta_1 * X))**2))

  #   # ax.scatter(theta_0, theta_1, np.min(residual_list), c='r', marker='o')
  #   ax.plot_surface(theta_0_grid, theta_1_grid, rss_vals, cmap='viridis')

  #   res_scatter = []
  #   for i in range(len(t_0)):
  #     res_scatter.append(np.sum((y- (t_0[i] + t_1[i] *X))**2))
  #   ax.scatter3D(t_0, t_1, res_scatter, c='r', marker='o')
  #   return fig
  def plot_surface(self, X, y, theta_0, theta_1):
    #def plot_surface(self, X, y, T0, T1):
    '''
    Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
    theta_0 and theta_1 over a range. Indicates the RSS based on given value of theta_0 and theta_1 by a
    red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.
      :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
      :param theta_0: Value of theta_0 for which to indicate RSS #pd Series of all theta_0
      :param theta_1: Value of theta_1 for which to indicate RSS #pd Series of all theta_1
      :return matplotlib figure plotting RSS
    '''
    coef0 = theta_0
    coef1 = theta_1
    theta_0 = 8
    theta_1 = 3 
    x = np.linspace(-1,1,50)
    y = theta_0 + theta_1*x
    print(theta_0, theta_1)
    def cost_func(theta_0, theta_1):
      theta_0 = np.atleast_3d(np.asarray(theta_0))
      theta_1 = np.atleast_3d(np.asarray(theta_1))
      return np.average((y -  (theta_0 + theta_1*x))**2)
    
    theta0_vals = np.linspace(self.coef_[0]-10, self.coef_[0]+10, 100)
    theta1_vals = np.linspace(self.coef_[1]-4, self.coef_[1]+4, 100)
    # theta0_grid, theta1_grid = np.meshgrid(theta0_vals, theta1_vals)
    # theta0_grid = np.linspace(min(self.coef_[0]-1,0),theta_0+1,101)
    # theta1_grid = np.linspace(min(theta_1-2,0),theta_1+2,101)
    b,m = np.meshgrid(theta0_vals, theta1_vals)
    zs = np.array([cost_func(bp,mp) for bp,mp in zip(np.ravel(b), np.ravel(m))])
    Z = zs.reshape(m.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(b, m, Z, rstride=1, cstride=1, cmap=cm.coolwarm, alpha=0.7)
    k = cost_func(coef0, coef1)
    ax.scatter([coef0],[coef1],[k], c='r', s=25, marker='.')
    ax.set_xlabel('theta_0')
    ax.set_ylabel('theta_1')
    ax.set_zlabel('Error')
    plt.title("Error:"+str(k))
    plt.show()
    return fig

  def plot_line_fit(self, X, y, theta_0, theta_1):
    """
    Function to plot fit of the line (y vs. X plot) based on chosen value of theta_0, theta_1. Plot must
    indicate theta_0 and theta_1 as the title.
      :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
      :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
      :param theta_0: Value of theta_0 for which to plot the fit
      :param theta_1: Value of theta_1 for which to plot the fit
      :return matplotlib figure plotting line fit
    """
    y_pred = theta_0 + theta_1*X

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot the actual y values and predicted y values as a line
    ax.plot(X, y, 'o', label='actual')
    ax.plot(X, y_pred, 'r-', label='predicted')

    # Set the title and legend
    ax.set_title(f'Line Fit (theta_0={theta_0:.2f}, theta_1={theta_1:.2f})')
    ax.legend()

    # Return the figure object
    plt.show()
    return fig


  def plot_contour(self, X, y, theta_0, theta_1):
    """
    Plots the RSS as a contour plot. A contour plot is obtained by varying
    theta_0 and theta_1 over a range. Indicates the RSS based on given value of theta_0 and theta_1, and the
    direction of gradient steps. Uses self.coef_ to calculate RSS.
      :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
      :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
      :param theta_0: Value of theta_0 for which to plot the fit
      :param theta_1: Value of theta_1 for which to plot the fit
      :return matplotlib figure plotting the contour
    """
        # Define a grid of theta_0 and theta_1 values to evaluate the RSS on
    theta_0_grid, theta_1_grid = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    
    # Compute the RSS for each combination of theta_0 and theta_1
    RSS_grid = np.zeros(theta_0_grid.shape)
    for i in range(theta_0_grid.shape[0]):
        for j in range(theta_0_grid.shape[1]):
            RSS_grid[i, j] = np.mean((y - theta_0_grid[i, j] - theta_1_grid[i, j]*X)**2)
    
    # Create a figure and axis object
    fig, ax = plt.subplots()
    
    # Plot the contour lines for the RSS
    ax.contour(theta_0_grid, theta_1_grid, RSS_grid, levels=np.logspace(-1, 3, 10))
    
    # Plot the starting point as a red dot
    ax.plot(theta_0, theta_1, 'ro', markersize=10)
    
    # Set the title and axis labels
    ax.set_title('RSS Contour Plot')
    ax.set_xlabel('theta_0')
    ax.set_ylabel('theta_1')
    
    plt.show()
    # Return the figure object
    return fig