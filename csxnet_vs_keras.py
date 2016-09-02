import time

from csxdata import CData, roots
from csxdata.utilities.parsers import mnist_tolearningtable as mnist_to_lt

datapath = roots["misc"] + "mnist.pkl.gz"
data = CData(mnist_to_lt(datapath, fold=False), cross_val=.2, header=False, standardize=True)


def run_csxnet():
    start = time.time()
    from csxnet.model import Network
    nw = Network(data, 0.3, 0.0, 0.0, 0.0, "xent")
    nw.name = "CsxNet Candidate"
    nw.add_fc(120)
    nw.finalize_architecture()
    nw.learn(batch_size=10, epochs=30)
    end = time.time()
    tcost, tacc = nw.evaluate("testing", accuracy=True)
    print("Final CsxNet accuracy on testing data: {}".format(tacc))
    print("Run time was {} seconds.".format(end - start))
    return nw, tacc, end - start


def run_keras():
    start = time.time()
    from keras.models import Sequential
    from keras.layers.core import Dense
    from keras.optimizers import SGD

    nw = Sequential()
    nw.name = "Keras Candidate"
    nw.add(Dense(input_dim=784, output_dim=120, activation="tanh"))
    nw.add(Dense(output_dim=10, activation="sigmoid"))
    nw.compile(optimizer=SGD(0.3), loss="categorical_crossentropy", metrics=["acc"])

    X, y = data.table("learning")
    tX, ty = data.table("testing")
    nw.fit(X, y, batch_size=10, nb_epoch=30, validation_data=(tX, ty))
    end = time.time()
    tcost, tacc = nw.evaluate(tX, ty)
    print("Final Keras accuracy on testing data: {}".format(tacc))
    print("Run time was {} seconds.".format(end - start))
    return nw, tacc, end - start


def compare(params1, params2):
    model1, acc1, time1 = params1
    model2, acc2, time2 = params2
    accw = int(acc2 > acc1)
    timew = int(time2 > time2)
    names = model1.name, model2.name
    accs = acc1, acc2
    times = time1, time2
    winner = names[accw]
    print("{} was more accurate!:\n{}:\t{}\n{}:\t{}".format(winner, names[0], accs[0], names[1], accs[1]))
    winner = names[timew]
    print("{} was more faster!:\n{}:\t{}\n{}:\t{}".format(winner, names[0], times[0], names[1], times[1]))


if __name__ == '__main__':
    compare(run_csxnet(), run_keras())
