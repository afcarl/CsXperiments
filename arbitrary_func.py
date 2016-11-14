import numpy as np
from keras.layers import Dense, Highway

from keras.models import Sequential as Network

from csxdata import RData
from keras.optimizers import SGD

X = np.random.uniform(-360.0, 360.0, size=(100000, 1))
y = np.sin(X)

data = RData((X, y), cross_val=0.1, indeps_n=1, header=None)
data.transformation = "std"

net = Network([
    Dense(300, activation="tanh", input_dim=data.neurons_required[0]),
    Dense(data.neurons_required[1], activation="tanh")
])
net.compile(SGD(lr=0.001, momentum=0.9), loss="mse")

net.fit(X, y, batch_size=20, validation_split=0.1)

test = np.arange(0, 361, 45, dtype=float)[:, None]
preds = net.predict(test, verbose=0)
preds = np.round(preds, 4)
real = np.sin(test)
real = np.round(real, 4)

for tx, ptx, ty in zip(test.ravel(), preds.ravel(), real.ravel()):
    print("sin({}) = {} (real: {})".format(tx, ptx, ty))
