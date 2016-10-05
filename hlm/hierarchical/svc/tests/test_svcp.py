from hlm.hierarchical.svcp import SVCP
from hlm.abstracts import Trace
from hlm._constants import TEST_SEED, RTOL, ATOL
from hlm.utils import no_op
import unittest as ut
import pandas as pd
import pysal as ps
import numpy as np
import os


FULL_PATH = os.path.dirname(os.path.abspath(__file__))

class Test_SVCP(ut.TestCase):
    def setUp(self):
        
        self.answer = Trace.from_csv(FULL_PATH + '/data/svcp.csv')
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
    
    def test_draw(self):
        instance = SVCP(**self.inputs, n_samples=0)
        np.random.seed(TEST_SEED)
        instance.draw()
        instance.trace._assert_allclose(self.answer,
                                        rtol=RTOL, atol=ATOL,
                                        ignore_shape = self.ignore_shape,
                                        squeeze=False)
