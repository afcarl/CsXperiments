import numpy as np
from numba import jit


@jit(nopython=True)
def sigmoid(z):
    return 1. / (1. + np.exp(z))


@jit(nopython=True)
def softmax(z):
    eX = np.exp(z)
    return eX / eX.sum(axis=1)


@jit(nopython=True)
def dense(x, w, b):
    return x @ w + b

