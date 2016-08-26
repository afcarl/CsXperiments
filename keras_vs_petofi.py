from keras.models import Sequential
from keras.layers import LSTM, Dense

from csxdata import Sequence, roots
from csxdata.utilities.helpers import speak_to_me


def pull_data():
    return Sequence(roots["txt"] + "petofi.txt", embeddim=10, cross_val=0.0, timestep=5)


def get_net():
    model = Sequential([
        LSTM(120, input_dim=(5, 10)),
        Dense(10)
    ])
    model.compile("adagrad", "mse")
    return model


def xperiment():
    petofi = pull_data()
    model = get_net()

    X, y = petofi.table("learning")
    speak_to_me(model, petofi)

    for century in range(1, 10):
        model.fit(X, y, nb_epoch=100)
        speak_to_me(model, petofi)

if __name__ == '__main__':
    xperiment()
