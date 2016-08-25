"""
Backstory:

I tried to implement a single linear neuron to do linear regression.
And it just won't work! I couldn't get it to work with NumPy. I thought
I messed up the derivatives so I moved to Theano, but it still wouldn't
work. I finally moved to Keras to assert if it was only my stupidity,
but it still wouldn't converge, unless I replaced the linear activation
function with sigmoid. Why is this??? I want to know now :(
"""


import numpy as np


def run_on_numpy():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [1]])
    W = np.random.randn(2, 1) * np.sqrt(2)
    b = 0.0
    eta = .1

    for epoch in range(1, 11):
        output = X.dot(W) + b
        error = y - output
        delta = X.T.dot(error)
        W -= eta * delta
        b -= error.sum()


def run_on_theano():
    import time
    import theano
    import theano.tensor as T

    X = T.matrix(name="X")
    y = T.matrix(name="Y")
    W = theano.shared(np.random.randn(2, 1), name="W")
    b = theano.shared(0.0, name="b")

    pred = X.dot(W) + b
    cost = T.sum((y - pred) ** 2)

    fit = theano.function(inputs=[X, y],
                          updates=[(W, W - T.grad(cost, wrt=W)),
                                   (b, b - T.grad(cost, wrt=b))],
                          outputs=[cost, pred])

    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype("float32")
    targets = np.array([[0], [1], [1], [1]]).astype("float32")

    for epoch in range(1, 11):
        c, p = fit(inputs, targets)
        print("-" * 20)
        print("Epoch", epoch)
        print("Cost:", c)
        print("O:", p.T)
        print("y:", targets.T)
        time.sleep(0.5)


def run_on_keras():
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import SGD

    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype("float32")
    targets = np.array([[0], [1], [1], [1]]).astype("float32")

    model = Sequential()
    model.add(Dense(input_dim=2, output_dim=1, activation="sigmoid"))
    model.compile(optimizer=SGD(lr=1.0), loss="mse", metrics=["accuracy"])

    model.fit(inputs, targets, batch_size=4, nb_epoch=100, verbose=1)
