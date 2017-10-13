import numpy as np

from csxdata.utilities.vectorop import split_by_categories


def _calc_scatters(X, Y):
    Xs = split_by_categories(Y, X)
    N = np.array([len(x) for x in Xs.values()])
    M = np.array([x.mean(axis=0) for x in Xs.values()])
    mu = M.mean(axis=0)
    Mc = np.array([m - mu for m in M])
    Xc = np.concatenate([x - m for x, m in zip(Xs.values(), M)], axis=0)
    Sw = (Xc.T @ Xc) / len(X)
    Sb = (Mc.T @ Mc) * N / len(X)

    return Sw, Sb


def manova(X, Y, p=0.05, stat="wilk"):
    Sw, Sb = _calc_scatters(X, Y)
    W = np.linalg.eigvals(np.linalg.inv(Sw) @ Sb)
    Wilks = np.prod(1. / (1. + W))


if __name__ == '__main__':
    np.random.seed(1337)
    tX = np.random.randn(120, 3).astype(float)
    tY = np.concatenate([np.full(40, i) for i in range(1, 4)])
    _calc_scatters(tX, tY)
