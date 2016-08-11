import time

from csxnet.utilities.pure import roots
from csxnet.data import CData, mnist_to_lt

datapath = roots["misc"] + "mnist.pkl.gz"
data = CData(mnist_to_lt(datapath, reshape=False), cross_val=.2, header=False, pca=0)
data.self_standardize()


def time_csxnet():
    start = time.time()
    from csxnet.model import Network
    nw = Network(data, 0.3, 0.0, 0.0, 0.0, "xent")
    nw.add_fc(120)
    nw.finalize_architecture()
    for epoch in range(1, 31):
        nw.learn(10)
        print("Epoch {} done!".format(epoch))
    tacc = nw.evaluate("testing")
    print("Final CsxNet accuracy on testing data: {}".format(tacc))
    print("Run time was {} seconds.".format(time.time() - start))


def time_keras():
    start = time.time()
    from keras.models import Sequential
    from keras.layers.core import Dense
    from keras.optimizers import SGD

    nw = Sequential()
    nw.add(Dense(input_dim=784, output_dim=120, activation="tanh"))
    nw.add(Dense(output_dim=10, activation="sigmoid"))
    nw.compile(optimizer=SGD(0.3), loss="categorical_crossentropy")

    X, y = data.table("learning")
    tX, ty = data.table("testing")
    nw.fit(X, y, batch_size=10, nb_epoch=30)
    tacc = nw.test_on_batch(tX, ty, accuracy=True)
    print("Final Keras accuracy on testing data: {}".format(tacc))
    print("Run time was {} seconds.".format(time.time() - start))

if __name__ == '__main__':
    time_csxnet()
    time_keras()
