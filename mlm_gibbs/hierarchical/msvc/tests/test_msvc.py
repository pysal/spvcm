from mlm_gibbs.hierarchical.msvc import MSVC
from mlm_gibbs.abstracts import Trace
from mlm_gibbs.tests.utils import Model_Mixin
from mlm_gibbs.utils import no_op
from mlm_gibbs._constants import TEST_SEED, RTOL, ATOL
import unittest as ut
import pandas as pd
import pysal as ps
import numpy as np
import os


FULL_PATH = os.path.dirname(os.path.abspath(__file__))

class Test_MSVC(ut.TestCase):
    def setUp(self):

        self.answer = Trace.from_csv(FULL_PATH + '/data/msvc.csv')
        self.inputs = dict()
        baltim = ps.pdio.read_files(ps.examples.get_path('baltim.shp'))
        Y = np.log(baltim.PRICE.values).reshape(-1,1)
        Yz = Y - Y.mean()
        X = baltim[['AGE', 'LOTSZ', 'SQFT']].values
        Xz = X-X.mean(axis=0)
        coords = baltim[['X', 'Y']].values
        self.inputs.update({'Y':Yz, 'X':Xz, 'coordinates':coords})
        self.ignore_shape = True
        self.test_trace = no_op
        self.inputs['configs'] = dict(jump=.5)

    def test_draw(self):
        self.inputs['n_samples'] = 0
        instance = MSVC(**self.inputs)
        np.random.seed(TEST_SEED)
        instance.draw()
        instance.trace._assert_allclose(self.answer,
                                        rtol=RTOL, atol=ATOL,
                                        ignore_shape = self.ignore_shape,
                                        squeeze=False)
