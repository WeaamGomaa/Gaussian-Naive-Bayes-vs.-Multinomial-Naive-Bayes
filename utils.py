import numpy as np


def compute_mean(x):
    return sum(x) / len(x)


def compute_var(x):
    m = compute_mean(x)
    return sum((xi - m) ** 2 for xi in x) / len(x)


def compute_accuracy(y_actual, y_predicted):
    return np.sum(y_actual == y_predicted) / len(y_actual)
