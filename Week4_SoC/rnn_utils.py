import numpy as np


def softmax(x):
    """Numerically-stable softmax over axis 0 (the class/feature axis)."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    """Element-wise logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))
