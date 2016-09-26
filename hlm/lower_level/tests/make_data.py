from hlm._constants import TEST_SEED, CLASSTYPES
from hlm.tests.utils import run_with_seed
from hlm import lower_level as M
from hlm.abstracts import Sampler_Mixin
from hlm.utils import south
import pandas as pd
import os

FULL_PATH = os.path.dirname(os.path.abspath(__file__))

def _build_data():
    models = []
    for cand in M.__dict__.values():
        if isinstance(cand, CLASSTYPES):
            if issubclass(cand, Sampler_Mixin):
                models.append(cand)
    for model in models:
        env = south()
        del env['M']
        run_with_seed(model, env=env, seed=TEST_SEED, fprefix=FULL_PATH + '/data/')
    return os.listdir(FULL_PATH + '/data/')
