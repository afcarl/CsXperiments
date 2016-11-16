import numpy as np
from numba import jit


@jit("f8[:,:](f8[:,:],f8[:,:],f8[:])", nopython=True)
def forward_tanh(X, W, b):
    return np.tanh(X.dot(W) + b)


@jit("f8[:,:](f8[:,:],f8[:,:])")
def nablaw(delta, inputs):
    return inputs.T.dot(delta)


@jit("f8[:,:](f8[:,:],f8[:,:])")
def nablax(delta, W):
    return delta.dot(W.T)


@jit("f8(f8[:,:],f8[:,:]", locals={"N": "int8"})
def mse(Y, T):
    return ((T-Y) ** 2.).sum() / 0.5 * Y.shape[0]


@jit("f8[:,:](f8[:,:])")
def sigmoid(Z):
    return 1. / (1. + np.exp(-Z))


@jit("f8[:,:](f8[:,:])")
def sigmoid_prime(A):
    return A * (1. - A)


@jit("f8[:,:](f8[:,:])")
def tanh_prime(A):
    return 1. - A ** 2.

@jit()
def feedforward(X, W, b, activation):
    return activation(X.dot(W) + b)

