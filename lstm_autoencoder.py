from csxdata import Sequence
from csxdata import roots
from csxdata.utilities.helpers import speak_to_me
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping


petofi = Sequence(roots["txt"] + "petofi.txt", cross_val=0.2, timestep=10)


def get_lstm():
    inshape, outputs = petofi.neurons_required
    model = Sequential([
        LSTM(input_shape=inshape, output_dim=120, return_sequences=True),
        LSTM(output_dim=120),
        Dense(output_dim=outputs, activation="softmax")
    ])
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
    return model


def xperimet():
    X, y = petofi.table("learning")
    valdat = petofi.table("testing")
    model = get_lstm()
    model.fit(X, y, callbacks=[EarlyStopping()], validation_data=valdat)

    print(speak_to_me(model, dat=petofi))


if __name__ == '__main__':
    xperimet()
