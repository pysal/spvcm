from hlm._constants import TEST_SEED, CLASSTYPES
from ... import both_levels as M
from hlm.tests.utils import run_with_seed
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
        run_with_seed(model, seed=TEST_SEED, fprefix=FULL_PATH + '/data/')
    return os.listdir(FULL_PATH + '/data/')
