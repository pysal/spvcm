from hlm.hierarchical.svcp import SVCP
from hlm._constants import TEST_SEED
import pysal as ps
import numpy as np
import os

FULL_PATH = os.path.dirname(os.path.abspath(__file__))

def _make_data():
    baltim = ps.pdio.read_files(ps.examples.get_path('baltim.shp'))
    coords = baltim[['X', 'Y']].values
    Y = np.log(baltim.PRICE.values).reshape(-1,1)
    Yz = Y - Y.mean()
    X = baltim[['AGE', 'LOTSZ', 'SQFT']].values
    Xz = X-X.mean(axis=0)
    
    model = SVCP(Yz, Xz, coordinates=coords, n_samples=0, phi_jump=.5)
    np.random.seed(TEST_SEED)
    model.draw()
    model.trace.to_df().to_csv(FULL_PATH + '/data/svcp.csv')

if __name__ == '__main__':
    _make_data()
