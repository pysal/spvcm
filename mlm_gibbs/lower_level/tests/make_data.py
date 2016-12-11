from mlm_gibbs._constants import TEST_SEED, CLASSTYPES
from mlm_gibbs.tests.utils import run_with_seed
from mlm_gibbs import lower_level as M
from mlm_gibbs.abstracts import Sampler_Mixin
from mlm_gibbs.utils import south
import pandas as pd
import os

FULL_PATH = os.path.dirname(os.path.abspath(__file__))

def build():
    models = []
    for cand in M.__dict__.values():
        if isinstance(cand, CLASSTYPES):
            if issubclass(cand, Sampler_Mixin):
                models.append(cand)
    for model in models:
        print('starting {}'.format(model))
        env = south()
        del env['M']
        run_with_seed(model, env=env, seed=TEST_SEED, fprefix=FULL_PATH + '/data/')
    return os.listdir(FULL_PATH + '/data/')
