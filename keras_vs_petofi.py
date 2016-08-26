from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import *

from csxdata import Sequence, roots, log
from csxdata.utilities.helpers import speak_to_me



def pull_data():
    return Sequence(roots["txt"] + "petofi.txt", embeddim=None, cross_val=0.0, timestep=10)


def get_net(inshape, outputs):
    model = Sequential([
        LSTM(180, input_shape=inshape, activation="tanh"),
        Dense(60, activation="tanh"),
        Dense(outputs, activation="softmax")
    ])
    model.compile(Adagrad(), "categorical_crossentropy")
    return model


def xperiment():
    log(" ----- ")
    petofi = pull_data()
    inshape, outputs = petofi.neurons_required
    model = get_net(inshape, outputs)

    X, y = petofi.table("learning")
    spoken = [speak_to_me(model, petofi)]
    log(spoken[0])
    print(spoken[0])

    for century in range(1, 20):
        model.fit(X, y, nb_epoch=100)
        spoken.append("Epoch {}: {}".format(century * 100, speak_to_me(model, petofi)))
        print(spoken[-1])
        log(spoken[-1])

    print("Run ended! Generated text:")
    print("\n".join(spoken))

if __name__ == '__main__':
    xperiment()
