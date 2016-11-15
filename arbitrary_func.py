import numpy as np

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.regularizers import WeightRegularizer


def batch_generator(m, N=None):
    """

    :param m: batch size
    :param N: overall number of samples to be drawn

    :returns: input m x 1 matrix and target m x 1 matrix
    """

    n = 0

    while 1:
        n += m
        if N:
            if n > N:
                m = N - m
                n = N

        inputs = np.random.uniform(-360., 360., size=(m, 1))
        targets = np.sin(inputs)

        yield inputs, targets
        if n == N:
            break

net = Sequential([
    Dense(120, activation="tanh", input_dim=1, W_regularizer=WeightRegularizer(l2=3.0)),
    Dense(120, activation="tanh", W_regularizer=WeightRegularizer(l2=3.0)),
    Dense(120, activation="tanh", W_regularizer=WeightRegularizer(l2=3.0)),
    Dense(60, activation="tanh", W_regularizer=WeightRegularizer(l2=3.0)),
    Dense(30, activation="tanh", W_regularizer=WeightRegularizer(l2=3.0)),
    Dense(1, activation="linear")
])
net.compile(SGD(lr=0.001), loss="mse")
net.fit_generator(generator=batch_generator(100), samples_per_epoch=100000, nb_epoch=100,
                  validation_data=batch_generator(50), nb_val_samples=1000)

test = np.arange(0, 361, 45, dtype=float)[:, None]
preds = net.predict(test, verbose=0)
preds = np.round(preds, 4)
real = np.sin(test)
real = np.round(real, 4)

for tx, ptx, ty in zip(test.ravel(), preds.ravel(), real.ravel()):
    print("sin({}) = {} (real: {})".format(tx, ptx, ty))
