from typing import Union
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd



# def accuracy(h_hat: pd.Series, y: pd.Series):
#     assert y_hat.size == y.size
#     correct_predictions = pd.Series(y_hat == y).value_counts()[True]
#     accuracy_ans = correct_predictions / y.size
#     return accuracy_ans

def accuracy(y_hat, y, print_sklearn = True):
        
    y_hat=pd.Series(y_hat, dtype="category")
    if print_sklearn:
        print("sklearn accuracy: ",accuracy_score(y,y_hat))
    return accuracy_score(y,y_hat)

# def precision(y_hat, y, cls):
#     y_hat=pd.Series(y_hat, dtype="category")  
#     equal_to_y_and_cls = pd.Series((y_hat == y) & (y_hat == cls)).value_counts()[True]
#     precision_ans = equal_to_y_and_cls / pd.Series(y_hat == cls).value_counts()[True]
#     return precision_ans

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str], print_sklearn = True) -> float:
    if print_sklearn: 
        print("sklearn precision: ",precision_score(y,y_hat,average=None)[cls])
    return precision_score(y,y_hat,average=None)[cls]



# def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    # equal_to_y_and_cls = pd.Series((y_hat == y) & (y_hat == cls)).value_counts()[True]
    # recall_ans = equal_to_y_and_cls / pd.Series(y == cls).value_counts()[True]
    # return recall_ans

def recall(y_hat, y, cls, print_sklearn = True):
    y_hat=pd.Series(y_hat, dtype="category")
    if print_sklearn:
        print("sklearn recall: ",recall_score(y,y_hat,average=None)[cls])
    return recall_score(y,y_hat,average=None)[cls]



# def rmse(y_hat: pd.Series, y: pd.Series) -> float:
#     error = y-y_hat
#     square_err = error**2
#     mean_square_error = square_err.mean()
#     rmse = mean_square_error**0.5
#     return rmse

def rmse(y_hat, y):
    y_hat=pd.Series(y_hat)
    print("sklearn rmse: ",mean_squared_error(y,y_hat)**0.5)
    return mean_squared_error(y,y_hat)**0.5


def mse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    error = y-y_hat
    square_err = error**2
    mean_square_error = square_err.mean()
    return mean_square_error
    # return rmse


# def mae(y_hat: pd.Series, y: pd.Series) -> float:
#     #find mae using sklearn
#     y_hat=pd.Series(y_hat)
#     print("sklearn mae: ",mean_absolute_error(y,y_hat))

#     """
#     Function to calculate the mean-absolute-error(mae)
#     """
#     abs_error = abs(y-y_hat)
#     mean_abs_error = abs_error.mean()
#     return mean_abs_error

def mae(y_hat, y):
    return mean_absolute_error(y,y_hat)

# def accuracy(y_hat, y):
#     """
#     Function to calculate the accuracy

#     Inputs:
#     > y_hat: pd.Series of predictions
#     > y: pd.Series of ground truth
#     Output:
#     > Returns the accuracy as float
#     """
#     """
#     The following assert checks if sizes of y_hat and y are equal.
#     Students are required to add appropriate assert checks at places to
#     ensure that the function does not fail in corner cases.
#     """
#     assert(y_hat.size == y.size)
#     # TODO: Write here
#     pass

# def precision(y_hat, y, cls):
#     """
#     Function to calculate the precision

#     Inputs:
#     > y_hat: pd.Series of predictions
#     > y: pd.Series of ground truth
#     > cls: The class chosen
#     Output:
#     > Returns the precision as float
#     """
#     pass

# def recall(y_hat, y, cls):
#     """
#     Function to calculate the recall

#     Inputs:
#     > y_hat: pd.Series of predictions
#     > y: pd.Series of ground truth
#     > cls: The class chosen
#     Output:
#     > Returns the recall as float
#     """
#     pass

# def rmse(y_hat, y):
#     """
#     Function to calculate the root-mean-squared-error(rmse)

#     Inputs:
#     > y_hat: pd.Series of predictions
#     > y: pd.Series of ground truth
#     Output:
#     > Returns the rmse as float
#     """

#     pass

# def mae(y_hat, y):
#     """
#     Function to calculate the mean-absolute-error(mae)

#     Inputs:
#     > y_hat: pd.Series of predictions
#     > y: pd.Series of ground truth
#     Output:
#     > Returns the mae as float
#     """
#     pass
