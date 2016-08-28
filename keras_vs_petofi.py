from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import *

from csxdata import Sequence, roots, log
from csxdata.utilities.helpers import speak_to_me


# params:
NGRAM = 1
SAMPLE_NO_NGRAMS = 50
TIMESTEP = 15
CROSSVAL = 0.0

PRETRAIN = RMSprop
PRETRAIN_LR = 0.001
FINETUNE = Adagrad
FINETUNE_LR = 0.01


def pull_data(crossval):
    return Sequence(roots["txt"] + "books.txt", embeddim=None, cross_val=crossval, timestep=TIMESTEP, n_gram=NGRAM)


def get_net(inshape, outputs):
    model = Sequential([
        LSTM(180, input_shape=inshape, activation="tanh"),
        Dense(60, activation="tanh"),
        Dense(outputs, activation="softmax")
    ])
    return model


def xperiment():
    log(" ----- Experiment: Keras VS Petőfi -----")

    def create_network_and_data(crossval=0.0):
        data = pull_data(crossval=crossval)
        inshape, outputs = data.neurons_required

        network = get_net(inshape, outputs)

        return network, data

    def pretrain_decade(decades):
        model.compile(PRETRAIN(lr=PRETRAIN_LR), "categorical_crossentropy")
        unbroken = True
        for decade in range(1, decades+1):
            try:
                model.fit_generator(lesson_generator, samples_per_epoch=32, nb_epoch=10, validation_data=val)
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
                model.fit_generator(lesson_generator, samples_per_epoch=32, nb_epoch=10, validation_data=val)
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

    model, petofi = create_network_and_data(crossval=CROSSVAL)

    lesson_generator = petofi.batchgen(32)
    val = None, None
    spoken = [sample()]

    pretrain_decade(20)
    finetune_decade(50)

    print()
    print("-"*50)
    print("Run ended! Generated text:")
    print("\n".join(spoken))

    longsample = "\n".join(sample() for _ in range(20))
    print("\n" + "-" * 50 + "Long sample:\n" + longsample)


if __name__ == '__main__':
    xperiment()
