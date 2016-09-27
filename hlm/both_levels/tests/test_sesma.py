from hlm import both as M
from hlm.tests.utils import Model_Mixin
from hlm.abstracts import Trace
import unittest as ut
import pandas as pd
from .make_data import FULL_PATH

class Test_SESMA(Model_Mixin, ut.TestCase):
    def setUp(self):
        Model_Mixin.build_self(self)
        self.cls = M.SESMA
        self.instance = self.cls(**self.inputs, n_samples=0)
        self.answer_trace = Trace.from_csv(FULL_PATH + '/data/sesma.csv')