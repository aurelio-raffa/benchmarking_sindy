import numpy as np


def threshold(x: np.ndarray, lb: float = 0.0, ub: float = 1.0):
    lower_bound = np.where(x >= lb, x, lb)
    res = np.where(lower_bound <= ub, lower_bound, ub)
    return res


def thresholded_sqrt(x):
    return np.sqrt(threshold(x))


def thresholded_sqrt_name(x):
    return f'sqrt({x})'
