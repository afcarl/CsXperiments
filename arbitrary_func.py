import time

import numpy as np
from matplotlib import pyplot as plt

from brainforge.util import white
from brainforge.costs import cost_fns
from brainforge.ops.activations import Tanh


def get_data():
    raw = np.linspace(-360, 360, num=100)[:, None]
    inpt = raw - raw.max()
    inpt /= (raw.max() - raw.min())
    trgt = np.sin(raw)
    return inpt, trgt

Wh = white(2, 10)
bh = np.zeros((10,))

Wo = white(10, 1)
bo = np.zeros((1,))

tanh = Tanh()
mse = cost_fns["mse"](None)
step = 1
costs = []

E = 20
lE = len(str(E))
for epoch in range(1, E+1):
    h = np.zeros((step, 10))
    o = np.zeros((step, 1))
    do = np.copy(o)

    Xs, Ys = get_data()
    print("Epoch {:>{w}}/{}".format(epoch, E, w=lE))
    for bno, (X, Y) in enumerate(((Xs[start:start+step], Ys[start:start+step])
                                 for start in range(0, len(Xs), step)), start=1):
        Z = np.concatenate((X, o), axis=1)
        h = tanh(Z @ Wh + bh)
        o = tanh(h @ Wo + bo)

        costs.append(mse(o, Y))
        do += (mse.derivative(o, Y) * tanh.derivative(o))
        dh = tanh.derivative(h) * do.dot(Wo.T)
        dZ = dh.dot(Wh.T)
        do = dZ[:, -1:]

        Wo -= (h.T @ do)
        bo -= do.sum(axis=0)
        Wh -= (X.T @ dh)
        bh -= dh.sum(axis=0)

        print("\rCost @ {:>4}: {:>6.3f}".format(bno, np.mean(costs)), end="")
        time.sleep(0.01)

    print()

costs = np.array(costs)
axX = np.arange(1, len(costs)+1)

plt.title("Run dynamics")
plt.plot(axX, costs, "b-")
plt.plot(axX, costs, "ro")
plt.show()
