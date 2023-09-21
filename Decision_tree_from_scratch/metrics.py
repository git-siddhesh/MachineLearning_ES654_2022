from typing import Union
import pandas as pd


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
    # print(y_hat)
    # print(y)
    correct_predictions = pd.Series(y_hat == y).value_counts()[True]
    accuracy_ans = correct_predictions / y.size
    return accuracy_ans



def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:

    """
    Function to calculate the precision
    """
    equal_to_y_and_cls = pd.Series((y_hat == y) & (y_hat == cls)).value_counts()[True]
    precision_ans = equal_to_y_and_cls / pd.Series(y_hat == cls).value_counts()[True]
    return precision_ans


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    equal_to_y_and_cls = pd.Series((y_hat == y) & (y_hat == cls)).value_counts()[True]
    recall_ans = equal_to_y_and_cls / pd.Series(y == cls).value_counts()[True]
    return recall_ans


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    error = y-y_hat
    square_err = error**2
    mean_square_error = square_err.mean()
    rmse = mean_square_error**0.5
    return rmse


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    abs_error = abs(y-y_hat)
    mean_abs_error = abs_error.mean()
    return mean_abs_error
