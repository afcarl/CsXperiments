import numpy as np

from csxnet import Network
from csxdata import RData


X = np.random.uniform(-10.0, 10.0, size=(100000, 1))
y = np.sin(X)

data = RData((X, y), cross_val=0.1, indeps_n=1, header=None)
data.transformation = "std"

net = Network(data, 0.01, 0.0, 0.0, 0.0, "mse", "TestDenseNet")
net.add_fc(120)
net.finalize_architecture()

net.fit(20, 10)

