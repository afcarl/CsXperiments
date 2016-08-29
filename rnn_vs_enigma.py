"""
This is an attempt to fit a stacked LSTM neural network onto a continous stream of
characters.
It seems that stacking autoencoders, no matter how deep or how many hidden units it contains
doesn't help to fit the stream :(
Increasing the timestep is also futile...

Maybe training an LSTM to predict the next element in Y based on previous elments of Y
and simultaneously traning another network to predict Y based on previous elemens of X
would lead to victory

"""
import random

import numpy as np
from csxdata.utilities.features import OneHot
from enigma import Machine, alphabet
from keras.layers import Dense, LSTM, Flatten
from keras.models import Sequential
from keras.optimizers import SGD

TIMESTEP = 1
BSIZE = len(alphabet) * 2
INSHP = [TIMESTEP, len(alphabet)]
OUTSHP = len(alphabet)


def get_machine():
    rotors = [('ESOVPZJAYQUIRHXLNFTGKDCMWB', 'J', 'G'),
              ('AJDKSIRUXBLHWTMCQGZNPYFVOE', 'E', 'M'),
              ('VZBRGITYUPSDNHLXAWMJQOFECK', 'Z', 'Y')]
    reflector = 'YRUHQSLDPXNGOKMIEBFZCWVJAT'
    plug_board = [('D', 'N'), ('G', 'R'), ('I', 'S'), ('K', 'C'), ('Q', 'X'), ('T', 'M'), ('P', 'V'), ('H', 'Y'),
                  ('F', 'W'), ('B', 'J')]

    machine = Machine(rotors, reflector, plug_board)

    return machine


def build_rnn():
    model = Sequential([
        LSTM(input_shape=INSHP, output_dim=300, return_sequences=True),
        LSTM(input_shape=INSHP, output_dim=300, return_sequences=True),
        LSTM(input_shape=INSHP, output_dim=300, return_sequences=True),
        LSTM(input_shape=INSHP, output_dim=300),
        Dense(output_dim=OUTSHP, activation="softmax")
    ])
    model.compile(optimizer=SGD(lr=0.01, momentum=0.8, nesterov=1),
                  loss="categorical_crossentropy", metrics=["acc"])
    return model


def build_fcnn():
    model = Sequential(layers=[
        Flatten(input_shape=INSHP),
        Dense(output_dim=120, activation="sigmoid"),
        Dense(output_dim=120, activation="sigmoid"),
        Dense(output_dim=120, activation="sigmoid"),
        Dense(output_dim=OUTSHP, activation="softmax")
    ])
    model.compile(SGD(lr=0.01, momentum=0.8, nesterov=True), loss="categorical_crossentropy")
    return model


def enigma_generator(machine, timestep, bsize):
    embed = OneHot()
    embed.fit(alphabet)

    def create_windows(ar):
        windows = []  # ew
        for start in range(bsize):
            end = start + timestep
            windows.append(ar[start:end])
            start += 1
        return np.stack(windows)

    while 1:
        # given bsize = 10 and timestep = 3, we need an initial X stream of (timestep + bsize + 1):
        # [x00] [x01] [x02] [x03] [x04] [x05] [x06] [x07] [x08] [x09] [x10] [x11] [x12] [x13] <- extra 1
        #      because      [y00] [y01] [y02] ... y always lags timestep + 1 steps behind X
        # also the final y comes after the final timestep-window in X
        instream = "".join([random.choice(alphabet) for _ in range(timestep + bsize)])
        outstream = machine.encrypt(instream[timestep:])

        instream = np.array(list(instream))
        outstream = np.array(list(outstream))

        X = embed(instream)
        X = create_windows(X)
        y = embed(outstream)

        assert X.shape[0] == y.shape[0]

        yield X, y


def validation_data(machine, timestep, bsize):
    gen = enigma_generator(machine, timestep=timestep, bsize=bsize)
    return next(gen)


def lstm_xperiment():
    turing = build_rnn()

    enigma = get_machine()
    datastream = enigma_generator(enigma, TIMESTEP, bsize=BSIZE)
    validation = validation_data(enigma, TIMESTEP, bsize=1000)

    turing.fit_generator(datastream, samples_per_epoch=BSIZE*100, nb_epoch=10, validation_data=validation)

    return turing


def fcnn_xperiment():
    enigma = get_machine()
    X, y = validation_data(enigma, 1, 10000)
    turing = build_fcnn()

    turing.fit(X, y)


if __name__ == '__main__':
    fcnn_xperiment()
