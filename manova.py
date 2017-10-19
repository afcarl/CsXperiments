import numpy as np

from csxdata.utilities.vectorop import split_by_categories


def _split_by_categories(X, Y):
    Xs = split_by_categories(Y, X)
    Ns = np.array([len(x) for x in Xs.values()])
    Ms = np.array([x.mean(axis=0) for x in Xs.values()])
    return Xs, Ns, Ms



def _calc_scatters(Xs, Ms, Ns):
    N = sum(Ns)
    mu = Ms.mean(axis=0)
    Mc = np.array([m - mu for m in Ms])
    Xc = np.concatenate([x - m for x, m in zip(Xs.values(), Ms)], axis=0)
    Sw = (Xc.T @ Xc) / N
    Sb = (Mc.T @ Mc) * Ns / N
    return Sw, Sb


def _wilks_lambda_to_F(W, N, K, D):
    a = N - K - ((D - K + 2) / 2)
    b_denom = D**2 + (K - 1)**2 - 5
    if b_denom > 0:
        b_numer = (D**2 * (K - 1)**2 - 4)
        b = np.sqrt(b_numer / b_denom)
    else:
        b = 1
    c = 0.5 * 
    wilks = np.prod(1. / 1. + W)


def _calc_pillai_barlett_trace(W):
    return np.sum(W / (1. + W))


def _calc_hotelling_trace(W):
    return np.sum(W)


def _calc_roy_greatest_root(W):
    return np.max(W)


def manova(X, Y, p=0.05, stat="wilk"):
    Xs, Ns, Ms = _split_by_categories(X, Y)
    Sw, Sb = _calc_scatters(Xs, Ns, Ms)
    E = np.linalg.inv(Sw) @ Sb
    W, V = np.linalg.eig(E)



if __name__ == '__main__':
    np.random.seed(1337)
    tX = np.random.randn(120, 3).astype(float)
    tY = np.concatenate([np.full(40, i) for i in range(1, 4)])
    _calc_scatters(tX, tY)
