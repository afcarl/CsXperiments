from brainforge import Network
from brainforge.layers import LSTM, DenseLayer, RLayer

from csxdata import roots, log
from csxdata.frames import Sequence
from csxdata.utilities.helpers import speak_to_me

# params:
NGRAM = 1
SAMPLE_NO_NGRAMS = 50
TIMESTEP = 15
CROSSVAL = 0.0
BSIZE = 50

DATASET = roots["txt"] + "petofi.txt"
CODING = "utf-8"


def pull_data():
    data = Sequence(DATASET, embeddim=None, cross_val=CROSSVAL,
                    timestep=TIMESTEP, n_gram=NGRAM, coding=CODING)
    print("Pulled data with N = {}!".format(data.N))
    return data


def get_net(inshape, outputs):
    model = Network(inshape, [
        RLayer(180, activation="tanh"),
        DenseLayer(60, activation="tanh"),
        DenseLayer(outputs, activation="sigmoid")
    ])
    model.finalize("xent", "adam")
    return model


def xperiment():
    log(" ----- Experiment: Brainforge VS Petőfi -----")

    def create_network_and_data():
        data = pull_data()
        inshape, outputs = data.neurons_required

        network = get_net(inshape, outputs)

        return network, data

    def run_decade(decades):
        for decade in range(1, decades+1):
            model.fit(X, y, epochs=1, validation=val)
            spoken.append("SGD finetune epoch {}: {}".format(10 * decade, speak_to_me(model, petofi)))
            print(spoken[-1])
            log(spoken[-1])

    def sample(stochastic=False):
        smpl = speak_to_me(model, petofi, stochastic, ngrams=SAMPLE_NO_NGRAMS)
        log(smpl)
        print(smpl)
        return smpl

    model, petofi = create_network_and_data()

    X, y = petofi.table("learning")
    val = petofi.table("testing")
    spoken = [sample()]

    run_decade(100)

    print()
    print("-"*50)
    print("Run ended! Generated text:")
    print("\n".join(spoken))

    longsample = "\n".join(sample() for _ in range(20))
    print("\n" + "-" * 50 + "Long sample:\n" + longsample)


if __name__ == '__main__':
    xperiment()
