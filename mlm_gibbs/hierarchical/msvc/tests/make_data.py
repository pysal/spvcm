from mlm_gibbs.hierarchical.msvc import MSVC
from mlm_gibbs._constants import TEST_SEED
import pysal as ps
import numpy as np
import os

FULL_PATH = os.path.dirname(os.path.abspath(__file__))

def build():
    baltim = ps.pdio.read_files(ps.examples.get_path('baltim.shp'))
    coords = baltim[['X', 'Y']].values
    Y = np.log(baltim.PRICE.values).reshape(-1,1)
    Yz = Y - Y.mean()
    X = baltim[['AGE', 'LOTSZ', 'SQFT']].values
    Xz = X-X.mean(axis=0)

    model = MSVC(Yz, Xz, coordinates=coords, n_samples=0, configs=dict(jump=.5))
    np.random.seed(TEST_SEED)
    model.draw()
    model.trace.to_csv(FULL_PATH + '/data/msvc.csv')
