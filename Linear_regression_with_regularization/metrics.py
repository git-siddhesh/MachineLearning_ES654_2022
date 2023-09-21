from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here
    y_hat = list(y_hat)
    y = list(y)
    correct_pred = 0
    for i in range(len(y)):
        if y[i] == y_hat[i]:
            correct_pred +=1
    return (correct_pred/len(y)) * 100


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    y = list(y)
    y_hat = list(y_hat)
    true_class = 0
    total_pred_class =  0# y_hat.count(cls)
    for i in range(len(y)):
        if y_hat[i] == cls:
            if y[i] == cls:
                true_class +=1
            total_pred_class +=1
    if total_pred_class == 0:
        return 0
    return (true_class/total_pred_class)*100



def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    y = list(y)
    y_hat = list(y_hat)
    true_class = 0
    total_actual_class = 0 #y.count(cls)
    for i in range(len(y)):
        if y[i]==cls:
            if y_hat[i] == cls:
                true_class +=1
            total_actual_class+=1
    if total_actual_class == 0:
        return 0
    return (true_class/total_actual_class) * 100


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    y_hat = np.array(y_hat)
    y = np.array(y)
    return (((y_hat-y)**2).mean())**0.5
    


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    y_hat = np.array(y_hat)
    y = np.array(y)

    return (abs(np.array(y_hat)-np.array(y))).mean()

'''
def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    # TODO: Write here
    pass

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    pass

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    pass

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """

    pass

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    pass
'''