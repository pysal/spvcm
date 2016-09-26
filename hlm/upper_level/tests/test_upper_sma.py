from hlm import upper
from hlm import utils
from hlm.tests.utils import Model_Mixin 
import unittest as ut
import pandas as pd
import os

FULL_PATH = os.path.dirname(os.path.abspath(__file__))

class Test_Upper_SMA(Model_Mixin, ut.TestCase):
    def setUp(self):
        Model_Mixin.build_self(self)
        self.cls = upper.SMA
        del self.inputs["W"]
        instance = self.cls(**self.inputs, n_samples=0)
        self.answer_df = pd.read_csv(FULL_PATH + '/data/upper_sma.csv')
