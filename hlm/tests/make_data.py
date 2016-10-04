from hlm.utils import south
import hlm
from hlm._constants import TEST_SEED
from hlm.diagnostics import psrf
import numpy as np
import os

FULL_PATH = os.path.dirname(os.path.abspath(__file__))


def build():
    data = south()
    del data['W']
    del data['M']
    model = hlm.both.MVCM(**data, n_samples=0)
    np.random.seed(TEST_SEED)
    print('starting South 5000, njobs=4')
    model.sample(5000,n_jobs=4)
    print('starting PSRF')
    known_brooks = psrf(model)
    known_gr = psrf(model, method='original')
    
    import json
    with open(FULL_PATH + '/data/' + 'psrf_brooks.json', 'w') as brooks:
        json.dump(known_brooks, brooks)
    with open(FULL_PATH + '/data/' + 'psrf_gr.json', 'w') as gr:
        json.dump(known_gr, gr)
    model.trace.to_csv(FULL_PATH + '/data/' + 'south_mvcm_5000.csv')
    return ([FULL_PATH + '/data/' + 'psrf_{}.json'.format(k)
             for k in ['brooks', 'gr']] + [FULL_PATH + '/data/south_mvcm_5000.csv'])
