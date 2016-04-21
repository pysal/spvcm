import pysal as ps
import unittest
import numpy as np
from hlm import HSAR, Base_HSAR
from pysal.common import RTOL

RTOL *= 10

import os.path as path
DATA_PATH = path.dirname(path.abspath(__file__))
SEED = 8879

def _check_moments(vector, known_mean, known_var, ax=0, rtol=RTOL):
    """
    This is a helper function to assist in testing a vector or matrix of
    quantities for its moments along a given axis.
    """
    exp_mean = vector.mean(axis=ax)
    np.testing.assert_allclose(exp_mean, known_mean, rtol=RTOL)
    exp_var = vector.var(axis=ax)
    np.testing.assert_allclose(exp_var, known_var, rtol=RTOL)
    return True

class TestHSAR(unittest.TestCase):
    def setUp(self):
        data = ps.open(path.join(DATA_PATH, 'test.csv'))
        self.y = data.by_col_array('y')
        self.X = data.by_col_array('x')
        W = ps.open(path.join(DATA_PATH, 'w_lower.mtx')).read()
        M = ps.open(path.join(DATA_PATH, 'w_upper.mtx')).read()
        W.transform = M.transform = 'r'
        self.W = W
        self.M = M
        self.membership = data.by_col_array('county') - 1 

    def test_userchecks(self):
        """
        This checks to see if all of the UserWarnings are raised correctly. 
        """
        try:
            HSAR(self.y, np.hstack((np.ones_like(self.y), self.X)),
                     self.W, self.M, membership=self.membership)
        except UserWarning:
            pass
        try:
            HSAR(self.y, self.X[0:5], self.W, self.M, 
                 membership=self.membership)
        except UserWarning:
            pass
        try:
            HSAR(self.y, self.X, self.W, self.M)
        except UserWarning:
            pass
        try:
            # synthesize a Delta so we can test if passing it &
            # membership fails out. 
            bad_Delta = np.repeat(np.eye(85), self.W.n // self.M.n, axis=0)
            fillup = np.zeros((self.W.n - self.M.n * (self.W.n // self.M.n),self.M.n))
            bad_Delta = np.vstack((bad_Delta, fillup))
            HSAR(self.y, self.X, self.W, self.M, 
                     membership=self.membership, Delta=bad_Delta)
        except UserWarning:
            pass
        try:
            HSAR(self.y, self.X, self.W, self.M,
                     membership=self.membership, err_grid=(-1,1))
        except UserWarning:
            pass
        try:
            badgrid = np.zeros_like(self.X)
            badgrid.dump('tmp.np')
            HSAR(self.y, self.X, self.W, self.M, membership=self.membership,
                    err_gridfile = 'tmp.np')
        except UserWarning:
            pass
    
    def test_cycle(self):
        """
        This checks if one cycle of the HSAR estimates correctly. This test must
        pass for later tests to pass. 
        """
        np.random.seed(SEED)
        reg = HSAR(self.y, self.X, self.W, self.M, 
                   Z=None, membership=self.membership, cycles=0) 
        self.reg = reg
        np.testing.assert_allclose(reg._state.Betas.flatten(), [0,0])
        self.reg.cycle()
        assert self.reg.position == 0
        assert self.reg.steps == 6
        assert self.reg.cycles == 1
        known_Betas = [0.759166, -0.831721] 
        np.testing.assert_allclose(self.reg.front.Betas.flatten(), known_Betas, rtol=RTOL)
        kts = np.array([ 0.07188605,  0.20338361,  0.23697536, -0.27245089, -0.97476678,
                     -0.44582875,  0.48252344, -0.12625246,  1.47917609, -0.92353755,
                     -2.11891015,  0.9175649 ,  2.46004152, -2.38379921,  1.35840552,
                     -1.1317675 ,  0.97786087,  0.63093027,  2.06659983,  1.31722943,
                      0.95193814,  2.4348122 , -1.27385296, -0.43541769,  1.1443172 ,
                      0.6087204 ,  0.71972704,  1.14521668, -0.69905879, -1.33835612,
                      1.35620344,  0.61040485, -0.62583598, -0.95341406,  0.62319454,
                      1.43944915, -1.21549111, -0.26528646,  0.66721907,  0.55966361,
                      2.96217155, -2.36136293, -2.84357421,  0.83354332,  0.64677957,
                      0.45411558, -1.72364288, -0.49624609,  0.59707393, -0.78541243,
                      1.35793114,  0.35419668, -0.26595075, -0.87616981,  1.9915893 ,
                      0.71075481, -2.66826143,  1.06066945,  2.05608351,  0.23467446,
                      3.82395519, -0.61615729,  1.28243645, -1.34890325,  0.41485979,
                      0.84767392,  0.04841721, -0.254669  ,  0.91970229,  1.29527977,
                     -1.87489348,  0.9434672 ,  0.11615686,  2.80684031, -1.47220803,
                     -0.97316664,  0.62713811,  1.55801109, -0.55501121, -0.614867  ,
                      0.74842159,  1.84226711,  0.4061868 ,  1.59197989, -1.42800899])
        kts = kts.reshape(self.M.n, 1)
        np.testing.assert_allclose(self.reg.front.Thetas, kts, rtol=RTOL)
        known_Sigma_e = np.array([[ 0.71745145]])
        np.testing.assert_allclose(self.reg.front.Sigma_e, known_Sigma_e, rtol=RTOL)
        known_Sigma_u = np.array([[ 1.43610929]])
        np.testing.assert_allclose(self.reg.front.Sigma_u, known_Sigma_u, rtol=RTOL)
        known_SAC_Upper = 0.28
        np.testing.assert_allclose(self.reg.front.SAC_Upper, known_SAC_Upper,rtol=RTOL)
        known_SAC_Lower = 0.04
        np.testing.assert_allclose(self.reg.front.SAC_Lower, known_SAC_Lower,rtol=RTOL)

    def test_sample(self):
        """
        This checks the properties of a deterministic sample of 100 elements
        from an HSAR sampler.
        """
        np.random.seed(SEED)
        reg = HSAR(self.y, self.X, self.W, self.M, 
                   Z=None, membership=self.membership, cycles=100) 
        
        Betas = np.vstack(reg.trace.Betas)
        known_means = np.array([ 1.061335, -0.600401])
        known_vars = np.array([ 0.038825,  0.008509])
        _check_moments(Betas, known_means, known_vars, ax=0, rtol=RTOL)

        Thetas = np.hstack(reg.trace.Thetas)
        ktm =np.array([ 0.27319802, -0.29591875, -0.1609023 ,  0.12282312,  0.0232627 ,
                        0.1307544 , -0.04090181, -0.1634346 ,  0.01100175, -0.24240923,
                       -0.20678282,  0.06664329, -0.19781733,  0.01439168,  0.03840102,
                       -0.07034768,  0.03392502,  0.09687011,  0.15763153,  0.13434407,
                       -0.05264853, -0.23302311, -0.12871759, -0.04266758,  0.16536654,
                        0.20153521,  0.15705536,  0.03065612,  0.0679658 , -0.13666245,
                        0.08472183,  0.01422825, -0.04925817,  0.14158519, -0.16718146,
                        0.05616415,  0.00685257, -0.09516859, -0.01786942,  0.10901389,
                       -0.06083823, -0.08663585, -0.04467019,  0.13837052, -0.02679628,
                        0.25395289,  0.05069002,  0.15505292, -0.08250934,  0.03246495,
                       -0.00353333,  0.07757543, -0.03633259, -0.02234181, -0.01523925,
                        0.07842939,  0.11093579,  0.13651698, -0.13616481,  0.08205798,
                        0.19743549,  0.01089495, -0.07805868, -0.10138955,  0.13773042,
                        0.05209063, -0.10438234, -0.0895457 , -0.0980288 ,  0.00640255,
                        0.03251165, -0.06495626, -0.10296913, -0.03969451, -0.15723285,
                        0.07645836,  0.23859218,  0.17045232,  0.06566287,  0.04053915,
                       -0.06274   , -0.22559345,  0.11375882,  0.17504021,  0.1403545 ])
        ktv =np.array([ 0.02832462,  1.43279771,  2.16263866,  1.4669337 ,  1.76177188,
                        1.87201947,  1.81930062,  1.67549688,  1.3202937 ,  1.745599  ,
                        1.79251973,  2.00777162,  1.74036091,  1.91406178,  1.7269117 ,
                        1.45813993,  1.29979391,  1.2452422 ,  1.27053183,  1.59491157,
                        1.63914856,  1.49007272,  1.66271719,  1.92567682,  1.35736535,
                        1.56533239,  1.6165343 ,  1.23788591,  1.59427253,  1.82544422,
                        1.92931057,  1.73468393,  1.1994736 ,  1.78665752,  1.2194713 ,
                        1.92596239,  1.79546367,  1.16738506,  1.19071729,  1.4793265 ,
                        1.52532561,  2.47966443,  2.16054875,  2.04617038,  1.38107438,
                        1.87425749,  1.69634625,  1.71109444,  1.47621287,  1.26824244,
                        1.54426045,  1.64154816,  1.6633056 ,  1.4009269 ,  1.60798001,
                        1.7901727 ,  1.1529488 ,  1.54920077,  1.37899683,  1.61630318,
                        1.91243571,  1.24098181,  1.45388671,  1.84757024,  1.83216483,
                        1.32024574,  1.52465426,  1.61791205,  1.52831652,  1.28388429,
                        1.27081514,  1.11435594,  1.68686102,  1.45205922,  1.40119822,
                        1.90167592,  1.96984172,  1.71594322,  1.42751606,  1.44212662,
                        1.2071908 ,  1.37463785,  2.20678905,  1.70128303,  1.52902204])
        _check_moments(Thetas, ktm, ktv, ax=1,rtol=RTOL)
        
        Sigma_e = np.vstack(reg.trace.Sigma_e)
        known_mean = 0.69557471
        known_var = 0.01828676
        _check_moments(Sigma_e, known_mean, known_var, ax=None, rtol=RTOL)
        
        Sigma_u = np.vstack(reg.trace.Sigma_u)
        known_mean = 1.51613146
        known_var = 1.50911197
        _check_moments(Sigma_u, known_mean, known_var, ax=None, rtol=RTOL)
        
        SAC_Upper = np.vstack(reg.trace.SAC_Upper)
        known_mean = -0.153564356
        known_var =  0.120660562
        _check_moments(SAC_Upper, known_mean, known_var, ax=None, rtol=RTOL)
        
        SAC_Lower = np.vstack(reg.trace.SAC_Lower)
        known_mean = -0.01742574
        known_var =  0.00492406
        _check_moments(SAC_Lower, known_mean, known_var, ax=None, rtol=RTOL)

if __name__ == '__main__':
    unittest.main()
