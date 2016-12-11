from mlm_gibbs import upper_level as upper
from mlm_gibbs import utils
from mlm_gibbs.tests.utils import Model_Mixin
from mlm_gibbs.abstracts import Trace
import unittest as ut
import pandas as pd
import os

FULL_PATH = os.path.dirname(os.path.abspath(__file__))

class Test_Upper_SE(ut.TestCase, Model_Mixin):
    def setUp(self):
        super(Test_Upper_SE, self).build_self()
        self.cls = upper.SE
        del self.inputs["W"]
        self.inputs['n_samples'] = 0
        instance = self.cls(**self.inputs)
        self.answer_trace = Trace.from_csv(FULL_PATH + '/data/upper_se.csv')
