import pandas as pd

from csxdata import roots

pd.TimeSeries(open(roots["txt"] + "books.txt").read())
