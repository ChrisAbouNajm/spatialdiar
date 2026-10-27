import numpy as np


def angdiff(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)
