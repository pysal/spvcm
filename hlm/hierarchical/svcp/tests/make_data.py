from hlm.hierarchical.svcp import SVCP
from hlm._constants import TEST_SEED
from hlm.utils import baltim
import pysal as ps
import numpy as np
import os

FULL_PATH = os.path.dirname(os.path.abspath(__file__))

def _make_data():
    inputs = baltim() 
    inputs['n_samples'] = 0
    inputs['phi_jump'] = .5
    model = SVCP(**inputs)
    np.random.seed(TEST_SEED)
    model.draw()
    model.trace.to_df().to_csv(FULL_PATH + '/data/svcp.csv')

if __name__ == '__main__':
    _make_data()
