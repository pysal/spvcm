from hlm import both as M
from hlm.tests.utils import Model_Mixin
import unittest as ut
import pandas as pd
from .make_data import FULL_PATH

class Test_SMASE(Model_Mixin, ut.TestCase):
    def setUp(self):
        Model_Mixin.build_self(self)
        self.cls = M.SMASE
        self.instance = self.cls(**self.inputs, n_samples=0)
        self.answer_df = pd.read_csv(FULL_PATH + '/data/smase.csv')