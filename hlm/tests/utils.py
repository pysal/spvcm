import unittest as ut
from hlm import utils
from hlm._constants import RTOL, ATOL, TEST_SEED, CLASSTYPES
from hlm.hierarchical.svcp.tests.test_svcp import Test_SVCP
from warnings import warn
import numpy as np
import copy
import types

class Model_Mixin(object):
    def build_self(self):
        super(Model_Mixin, self).__init__()
        self.inputs = utils.south()
        self.__dict__.update(self.inputs)
        self.ignore_shape = False
        self.squeeze = True

    def test_trace(self):
        instance = self.cls(**self.inputs, n_samples=0)
        np.random.seed(TEST_SEED)
        instance.draw()
        instance.trace._assert_allclose(self.answer_trace,
                                        rtol=RTOL, atol=ATOL,
                                        ignore_shape = self.ignore_shape,
                                        squeeze=self.squeeze)
    
    @ut.skip
    def test_argument_parsing(self):
        #priors, initial values, etc.
        raise NotImplementedError
    
def run_with_seed(cls, env=utils.south(), seed=TEST_SEED, fprefix = ''):
    fname = str(cls).strip("'<>'").split('.')[-1].lower()
    model = cls(**env, n_samples=0)
    np.random.seed(TEST_SEED)
    model.draw()
    model.trace.to_df().to_csv(fprefix + fname + '.csv', index=False)
