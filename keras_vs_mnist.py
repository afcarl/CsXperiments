from keras.models import Sequential
from keras.layers.core import Dense

from csxnet.data import mnist_to_lt, CData
from csxnet.utilities.pure import roots

miscpath = roots["misc"] + "mnist.pkl.gz"


def get_fcnn():
    network = Sequential()
    network.add(Dense(120, activation="tanh", input_dim=784))
    network.add(Dense(10, activation="softmax"))
    network.compile(optimizer="sgd", loss="categorical_crossentropy")

    return network


def get_cnn():
    from keras.layers.convolutional import Convolution2D
    from keras.layers.convolutional import MaxPooling2D
    from keras.layers.core import Flatten

    network = Sequential()
    network.add(Convolution2D(3, 5, 5, activation="tanh", input_shape=(1, 28, 28)))
    network.add(MaxPooling2D())
    network.add(Flatten())
    network.add(Dense(60, activation="tanh"))
    network.add(Dense(10, activation="softmax"))
    network.compile(optimizer="sgd", loss="categorical_crossentropy")

    return network


def experiment(mode):
    mode = mode.lower()[0]
    if mode.lower()[0] == "c":
        chain = "Convolutional"
    elif mode.lower()[0] == "f":
        chain = "Fully Connected"
    else:
        raise RuntimeError("Wrong mode definition!")

    print("Experiment: MNIST classification with {} Neural Network!".format(chain))
    net = get_fcnn() if mode == "f" else get_cnn()
    mnist = CData(mnist_to_lt(miscpath, (True if mode == "c" else False)))
    mnist.self_standardize()

    net.fit(mnist.data, mnist.indeps, batch_size=20, nb_epoch=30, verbose=1,
            validation_split=0.2, show_accuracy=True)

if __name__ == '__main__':
    experiment("convolution")
    # experiment("fully connected")
