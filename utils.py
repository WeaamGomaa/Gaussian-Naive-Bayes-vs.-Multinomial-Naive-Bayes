"""
Temporary placeholder for Nagat's implementation.
Currently using NumPy functions for integration purposes.
"""
import numpy as np


def compute_mean(x):
    return np.mean(x)


def compute_var(x):
    return np.var(x)


def compute_accuracy(y_actual, y_predicted):
    return np.sum(y_actual == y_predicted) / len(y_actual)
