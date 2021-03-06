from keras.models import Sequential
from keras.layers import Dense

from csxdata import CData
from csxdata import roots


def pull_data(path):
    import pickle
    import gzip
    with gzip.open(path, "rb") as ltfl:
        lt = pickle.load(ltfl)
        ltfl.close()

    return CData(lt, .2)


def build_model():
    model = Sequential()
    model.add(Dense(input_dim=3600, output_dim=30))

pics = pull_data(roots["lt"] + "xonezero_bgs.pkl.gz")
pics.transformation = "std"

X, y = pics.table("learning")

