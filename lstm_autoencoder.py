from csxdata import MassiveSequence
from csxdata import roots
from csxdata.utilities.helpers import speak_to_me
from keras.layers import LSTM, Dense
from keras.models import Sequential


petofi = MassiveSequence(roots["txt"] + "petofi.txt", cross_val=0.2, timestep=10)


def get_lstm():
    inshape, outputs = petofi.neurons_required
    model = Sequential([
        LSTM(input_shape=inshape, output_dim=120),
        Dense(output_dim=outputs, activation="softmax")
    ])
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
    return model


def xperimet():
    X, y = petofi.table("learning")
    valdat = petofi.table("testing")
    model = get_lstm()
    model.fit(X, y, validation_data=valdat)

    print(speak_to_me(model, dat=petofi))


def batcher():
    generate = petofi.batchgen(10000)
    model = get_lstm()
    model.fit_generator(generate, samples_per_epoch=10000, nb_epoch=10)

    print(speak_to_me(model, dat=petofi))

if __name__ == '__main__':
    batcher()
