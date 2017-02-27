from keras.models import Model
from keras.layers import Dense, Input, Activation

from csxdata import CData, roots

mnist = CData(roots["misc"] + "mnist.pkl.gz", headers=None, cross_val=10000)
mnist.transformation = "std"

batchgen = mnist.batchgen(bsize=60, infinite=True)
inshape, outshape = mnist.neurons_required

inputs = Input(inshape)
stack = Dense(60, activation="tanh")(inputs)
stack = Dense(outshape[0])(stack)
output1 = Activation("softmax")(stack)
output2 = Activation("sigmoid")(stack)
model = Model(input=[inputs], output=[output1, output2])
model.compile(optimizer="rmsprop", loss="mse",
              loss_weights=[1., 1.])
model.fit_generator(batchgen, samples_per_epoch=mnist.N, nb_epoch=30,
                    validation_data=mnist.table("learning"))
