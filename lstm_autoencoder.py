import numpy as np

from keras.models import Model
from keras.layers import Input, LSTM, Dense, TimeDistributed
from keras.regularizers import l2


def build_encoder_part(inpt, layers):
    lstm = LSTM(layers["hidden1"], return_sequences=True)(inpt)
    dense = TimeDistributed(
        Dense(layers["hidden2"], activation="relu", kernel_regularizer=l2(0.01))
    )(lstm)
    encoder = Model(inputs=inpt, outputs=dense)
    return encoder


def build_decoder_part(encoder, layers):
    inpt_ = Input(batch_shape=encoder.output_shape)
    h1 = TimeDistributed(
        Dense(layers["hidden2"], activation="relu", kernel_regularizer=l2(0.01))
    )(inpt_)
    lstm = LSTM(layers["hidden1"], return_sequences=True)(h1)
    decoder_out = Dense(layers["input"])(lstm)
    decoder = Model(inputs=inpt_, outputs=decoder_out)
    return decoder


def assemble_autoencoder(sequence_length, layers):
    inpt = Input((sequence_length, layers["input"]))
    encoder = build_encoder_part(inpt, layers)
    decoder = build_decoder_part(encoder, layers)
    aetensor = decoder(inpt)
    aetensor = encoder(aetensor)
    autoencoder = Model(inpt, aetensor)
    autoencoder.compile(loss="mse", optimizer="adam")
    return autoencoder, encoder


N = 1000
T = 20
D = 32

X = np.random.randn(N, T, D)

aenc, enc = assemble_autoencoder(T, {"input": D, "hidden1": 20, "hidden2": 10})
aenc.fit(X, X)

encodedX = enc.predict(X)
