import unittest as ut
from hlm import utils
from hlm._constants import RTOL, ATOL, TEST_SEED, CLASSTYPES
import numpy as np
import copy
import types

class Model_Mixin(object):
    def build_self(self):
        self.inputs = utils.south()
        self.__dict__.update(self.inputs)

    def test_trace(self):
        instance = self.cls(**self.inputs, n_samples=0)
        np.random.seed(TEST_SEED)
        instance.draw()
        trace_df = instance.trace.to_df()
        for col in trace_df:
            np.testing.assert_allclose(trace_df[col].values,
                                       self.answer_df[col].values,
                                       rtol=RTOL, atol=ATOL,
                                       err_msg = 'Failed on {}'.format(col))

    def test_argument_parsing(self):
        #priors, initial values, etc.
        raise NotImplementedError
    
def run_with_seed(cls, env=utils.south(), seed=TEST_SEED, fprefix = ''):
    fname = str(cls).strip("'<>'").split('.')[-1].lower()
    model = cls(**env, n_samples=0)
    np.random.seed(TEST_SEED)
    model.draw()
    model.trace.to_df().to_csv(fprefix + fname + '.csv', index=False)
