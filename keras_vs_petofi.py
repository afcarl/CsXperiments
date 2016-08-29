from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import *

from csxdata import roots, log
from csxdata.frames import MassiveSequence, Sequence
from csxdata.utilities.helpers import speak_to_me


# params:
NGRAM = 1
SAMPLE_NO_NGRAMS = 50
TIMESTEP = 15
CROSSVAL = 0.0

BSIZE = 50
PRETRAIN = RMSprop
PRETRAIN_LR = 0.001
FINETUNE = Adagrad
FINETUNE_LR = 0.01

DATASET = roots["seq"] + "/homo_sapiens/chr1.fa"
# DATASET = roots["csvs"] + "reddit.csv"
# DATASET = roots["txt"] + "books.txt"
# DATASET = roots["txt"] + "petofi.txt"
CODING = "utf-8"


def pull_data(dframe):
    data = dframe(DATASET, embeddim=None, cross_val=CROSSVAL, timestep=TIMESTEP, n_gram=NGRAM, coding=CODING)
    print("Pulled data with N = {}!".format(data.N))
    return data


def get_net(inshape, outputs):
    model = Sequential([
        LSTM(180, input_shape=inshape, activation="tanh"),
        Dense(60, activation="tanh"),
        Dense(outputs, activation="softmax")
    ])
    return model


def xperiment():
    log(" ----- Experiment: Keras VS Petőfi -----")

    def create_network_and_data():
        data = pull_data(dframe=Sequence)
        inshape, outputs = data.neurons_required

        network = get_net(inshape, outputs)

        return network, data

    def pretrain_decade(decades):
        model.compile(PRETRAIN(lr=PRETRAIN_LR), "categorical_crossentropy")
        unbroken = True
        for decade in range(1, decades+1):
            try:
                model.fit(X, y, nb_epoch=10, validation_data=val)
            except KeyboardInterrupt:
                unbroken = False
            spoken.append("RMSprop pretrain epoch {}: {}".format(decade * 10, speak_to_me(model, petofi)))
            print(spoken[-1])
            log(spoken[-1])
            if not unbroken:
                log("RMSprop pretrain BROKEN with KEYBOARD_INTERRUPT")
                return

    def finetune_decade(decades):
        model.compile(FINETUNE(lr=FINETUNE_LR), "categorical_crossentropy")
        unbroken = True
        for decade in range(1, decades+1):
            try:
                model.fit(X, y, nb_epoch=10, validation_data=val)
            except KeyboardInterrupt:
                unbroken = False
            spoken.append("SGD finetune epoch {}: {}".format(10 * decade, speak_to_me(model, petofi)))
            print(spoken[-1])
            log(spoken[-1])
            if not unbroken:
                log("SGD finetune BROKEN with KEYBOARD_INTERRUPT")
                return

    def sample(stochastic=False):
        smpl = speak_to_me(model, petofi, stochastic, ngrams=SAMPLE_NO_NGRAMS)
        log(smpl)
        print(smpl)
        return smpl

    model, petofi = create_network_and_data()

    X, y = petofi.table("learning")
    val = petofi.table("testing")
    spoken = [sample()]

    pretrain_decade(5)
    finetune_decade(10)

    print()
    print("-"*50)
    print("Run ended! Generated text:")
    print("\n".join(spoken))

    longsample = "\n".join(sample() for _ in range(20))
    print("\n" + "-" * 50 + "Long sample:\n" + longsample)


def generators():
    log(" ----- Experiment: Keras VS Petőfi -----")

    def create_network_and_data():
        data = pull_data(dframe=MassiveSequence)
        inshape, outputs = data.neurons_required

        network = get_net(inshape, outputs)

        return network, data

    def pretrain_century(century):
        model.compile(PRETRAIN(lr=PRETRAIN_LR), "categorical_crossentropy")
        unbroken = True
        for decade in range(1, century+1):
            try:
                model.fit_generator(the_generator, samples_per_epoch=petofi.N, nb_epoch=10)
            except KeyboardInterrupt:
                unbroken = False
            spoken.append("RMSprop pretrain epoch {}: {}".format(decade * 10, speak_to_me(model, petofi)))
            print(spoken[-1])
            log(spoken[-1])
            if not unbroken:
                log("RMSprop pretrain BROKEN with KEYBOARD_INTERRUPT")
                return

    def finetune_century(century):
        model.compile(FINETUNE(lr=FINETUNE_LR), "categorical_crossentropy")
        unbroken = True
        for decade in range(1, century+1):
            try:
                model.fit_generator(the_generator, samples_per_epoch=petofi.N, nb_epoch=10)
            except KeyboardInterrupt:
                unbroken = False
            spoken.append("SGD finetune epoch {}: {}".format(10 * decade, speak_to_me(model, petofi)))
            print(spoken[-1])
            log(spoken[-1])
            if not unbroken:
                log("SGD finetune BROKEN with KEYBOARD_INTERRUPT")
                return

    def sample(stochastic=False):
        smpl = speak_to_me(model, petofi, stochastic, ngrams=SAMPLE_NO_NGRAMS)
        log(smpl)
        print(smpl)
        return smpl

    model, petofi = create_network_and_data()

    the_generator = petofi.batchgen(BSIZE)
    spoken = [sample()]

    pretrain_century(10)
    finetune_century(5)

    print()
    print("-"*50)
    print("Run ended! Generated text:")
    print("\n".join(spoken))

    longsample = "\n".join(sample() for _ in range(20))
    print("\n" + "-" * 50 + "Long sample:\n" + longsample)


if __name__ == '__main__':
    generators()
