import unittest as ut
import numpy as np
from hlm.utils import south
from hlm.both_levels import MVCM
from hlm.diagnostics import psrf
from hlm._constants import RTOL, ATOL, TEST_SEED
from hlm.abstracts import Trace, Hashmap
import os
import json
FULL_PATH = os.path.dirname(os.path.abspath(__file__))


class Test_PSRF(ut.TestCase):
    def setUp(self):
        data = south()
        data['n_samples'] = 0
        with open(FULL_PATH + '/data/psrf_brooks.json', 'r') as brooks:
            self.known_brooks = json.load(brooks)
        with open(FULL_PATH + '/data/psrf_gr.json', 'r') as gr:
            self.known_gr = json.load(gr)
        with open(FULL_PATH + '/data/psrf_noburn.json', 'r') as noburn:
            self.noburn = json.load(noburn)
        np.random.seed(TEST_SEED)
        self.trace = Trace.from_csv(FULL_PATH + '/data/south_mvcm_5000', multi=True)
        self.mockmodel = Hashmap(trace=self.trace)

    def test_coef_recovery(self):
        #test with:
        #model=model, trace=model.trace, chain=model.trace['asdf']
        #autoburnin=False, method='original'
        exp_brooks = psrf(self.mockmodel)
        for k,v in exp_brooks.items():
            if k == 'Alphas':
                continue
            np.testing.assert_allclose(v, self.known_brooks[k],
                                       rtol=RTOL, atol=ATOL,
                                       err_msg='Failed in {}'.format(k))
        exp_gr = psrf(trace=self.trace, method='original')
        for k,v in exp_gr.items():
            if k == 'Alphas':
                continue
            np.testing.assert_allclose(v, self.known_gr[k],
                                       rtol=RTOL, atol=ATOL,
                                       err_msg='Failed in {}'.format(k))
    
    def test_options(self):
        exp_brooks = psrf(trace=self.mockmodel.trace)
        for k,v in exp_brooks.items():
            if k == 'Alphas':
                continue
            np.testing.assert_allclose(v, self.known_brooks[k],
                                       rtol=RTOL, atol=ATOL,
                                       err_msg='Failed in {}'.format(k))
        exp_brooks = psrf(chain=self.mockmodel.trace['Tau2'])
        np.testing.assert_allclose(exp_brooks['parameter'],
                                   self.known_brooks['Tau2'],
                                   rtol=RTOL, atol=ATOL,
                                   err_msg='Failed in Tau2')
        test_completion = psrf(trace=self.trace, autoburnin=False)
        for k,v in self.noburn.items():
            if k == 'Alphas':
                continue
            np.testing.assert_allclose(v, test_completion[k],
                                       rtol=RTOL, atol=ATOL,
                                       err_msg='Failed in {}'.format(k))
        limit_vars = psrf(trace=self.trace, varnames=['Tau2', 'Sigma2'])
        for k,v in limit_vars.items():
            if k == 'Alphas':
                continue
            np.testing.assert_allclose(v, limit_vars[k],
                                       rtol=RTOL, atol=ATOL,
                                       err_msg='Failed in {}'.format(k))
        