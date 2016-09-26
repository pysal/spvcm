from hlm import lower
from hlm import utils
from hlm._constants import RTOL, ATOL, TEST_SEED, CLASSTYPES
from hlm.tests.utils import Model_Mixin, run_with_seed
from hlm.abstracts import Sampler_Mixin
import unittest as ut
import numpy as np
import pandas as pd
import os

FULL_PATH = os.path.dirname(os.path.abspath(__file__))

class Test_Lower_SE(Model_Mixin, ut.TestCase):
    def setUp(self):
        Model_Mixin.build_self(self)
        self.cls = lower.SE
        del self.inputs["M"]
        instance = self.cls(**self.inputs, n_samples=0)
        self.answer_df = pd.read_csv(FULL_PATH + '/data/lower_se.csv')
