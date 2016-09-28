from hlm import lower
from hlm.tests.utils import Model_Mixin
from hlm.abstracts import Trace
import unittest as ut
import pandas as pd
import os

FULL_PATH = os.path.dirname(os.path.abspath(__file__))

class Test_Lower_SMA(ut.TestCase, Model_Mixin):
    def setUp(self):
        super(Test_Lower_SMA, self).build_self()
        self.cls = lower.SMA
        del self.inputs["M"]
        instance = self.cls(**self.inputs, n_samples=0)
        self.answer_trace = Trace.from_csv(FULL_PATH + '/data/lower_sma.csv')
