from hlm.hierarchical.svcp import SVCP
from hlm.tests.utils import Model_Mixin
import unittest as ut
import pandas as pd
import pysal as ps
import numpy as np
import os


FULL_PATH = os.path.dirname(os.path.abspath(__file__))

class Test_SVCP(Model_Mixin, ut.TestCase):
    def setUp(self):
        Model_Mixin.build_self(self)
        del self.inputs

        self.inputs = dict()
        baltim = ps.pdio.read_files(ps.examples.get_path('baltim.shp'))
        Y = np.log(baltim.PRICE.values).reshape(-1,1)
        Yz = Y - Y.mean()
        X = baltim[['AGE', 'LOTSZ', 'SQFT']].values
        Xz = X-X.mean(axis=0)
        coords = baltim[['X', 'Y']].values
        self.inputs.update({'Y':Yz, 'X':Xz, 'coordinates':coords})
        
        self.cls = SVCP
        instance = self.cls(**self.inputs, n_samples=0)
        self.answer_trace = Trace.from_csv(FULL_PATH + '/data/svcp.csv')
