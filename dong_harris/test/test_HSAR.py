import pysal as ps
import unittest
import numpy as np
from pysal.spreg.hlm import HSAR, Base_HSAR
from pysal.common import RTOL

import os.path as path
DATA_PATH = path.dirname(path.abspath(__file__))

def _check_moments(vector, known_mean, known_var, ax=0, rtol=RTOL):
    """
    This is a helper function to assist in testing a vector or matrix of
    quantities for its moments along a given axis.
    """
    exp_mean = vector.mean(axis=ax)
    np.testing.assert_allclose(exp_mean, known_mean)
    exp_var = vector.var(axis=ax)
    np.testing.assert_allclose(exp_var, known_var)
    return True

class TestHSAR(unittest.TestCase):
    def setUp(self):
        np.random.seed(8879)
        data = ps.open(path.join(DATA_PATH, 'test.csv'))
        self.y = data.by_col_array('y')
        self.X = data.by_col_array('x')
        W = ps.open(path.join(DATA_PATH, './w_lower.mtx')).read()
        M = ps.open(path.join(DATA_PATH, './m_lower.mtx')).read()
        W.transform = M.transform = 'r'
        self.W = W
        self.M = M
        self.membership = data.by_col_array('county') - 1 

    def test_userchecks(self):
        """
        This checks to see if all of the UserWarnings are raised correctly. 
        """
        try:
            HSAR(self.y, np.hstack(np.ones_like(self.y), self.X),
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
                     membership=self.members, err_grid=(-1,1))
        except UserWarning:
            pass
        try:
            badgrid = np.zeros_like(self.X)
            badgrid.dump('tmp.np')
            HSAR(self.y, self.X, self.W< self.M, membership=self.membership,
                    err_gridfile = 'tmp.np')
        except UserWarning:
            pass
    
    def test_cycle(self):
        """
        This checks if one cycle of the HSAR estimates correctly. This test must
        pass for later tests to pass. 
        """
        reg = hlm.HSAR(self.y, self.X, self.W, self.M, 
                       Z=None, membership=self.membership, cycles=0) 
        self.reg = reg
        np.testing.assert_allclose(reg._state.Betas.flatten(), [0,0])
        self.reg.cycle()
        assert self.reg.position == 0
        assert self.reg.steps == 6
        assert self.reg.cycles == 1
        known_Betas = [.72387217, -.74392261] 
        np.testing.assert_allclose(self.reg.front.Betas.flatten(), known_Betas, rtol=RTOL)
        kts = array([ 0.07188605,  0.20338361,  0.23697536, -0.27245089, -0.97476678,
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
        self.reg.sample(cycles=100)
        
        Betas = np.vstack(self.reg.trace.Betas)
        known_means = np.array([ 0.987316  , -0.60069134])
        known_vars = array([ 0.11546268,  0.00843365])
        _check_moments(Betas, known_means, known_vars, known_vars, axis=0)

        Thetas = np.hstack(self.reg.Trace.Thetas)
        ktm =np.array([0.35138819, -0.04114799,  0.12224185,  0.02315829,  0.08836564,
                      -0.02478626, -0.08521599, -0.08403168, -0.08954327,  0.10310152,
                       0.01196054,  0.02825597,  0.17573173,  0.12661829,  0.01436851,
                      -0.016592  , -0.04098931, -0.14048582,  0.0160997 , -0.07494238,
                      -0.06121015, -0.12517103,  0.21984182, -0.23302946, -0.04787985,
                      -0.19375665, -0.05004593,  0.00311265, -0.02328693,  0.17230521,
                       0.11002931,  0.03902908, -0.18844595,  0.03631927,  0.03489406,
                       0.05869946,  0.00874079,  0.06807642,  0.11629303,  0.00409125,
                       0.12165762,  0.13821041,  0.05814636,  0.08270564, -0.19233712,
                       0.02555172, -0.04708598,  0.26233993, -0.22159353,  0.15985406,
                      -0.00037927,  0.22124527, -0.12617246,  0.01829536,  0.04817059,
                      -0.13739826,  0.08928174,  0.24820194,  0.01249968, -0.2237208 ,
                      -0.02893671,  0.10764536,  0.09927408, -0.26741299, -0.09298538,
                       0.08644144, -0.00671525,  0.03530498, -0.03568075,  0.02162343,
                       0.07822967, -0.12407675,  0.08830117,  0.04370886, -0.1664112 ,
                      -0.11591989,  0.12551598,  0.01197779, -0.14669674, -0.08492436,
                      -0.11428239,  0.00127711, -0.01361595,  0.11563549, -0.01709923])
        ktv =np.array([0.11025854,  1.58851862,  1.51230806,  1.37152659,  1.71094136,
                      1.63548062,  1.33046778,  2.14845035,  1.55424274,  1.37661106,
                      1.68718001,  1.71730531,  1.66255626,  1.66641851,  1.36438125,
                      1.31141907,  2.0443225 ,  1.61237473,  1.89019618,  1.52510202,
                      1.6667302 ,  1.44660315,  1.73310947,  1.75712202,  1.79130377,
                      1.82650495,  1.77426292,  0.98899998,  1.69306317,  1.83653261,
                      2.29892005,  1.82534464,  1.33395427,  1.77492753,  1.83856326,
                      1.31599694,  1.58147309,  1.81242396,  1.31459371,  1.76436165,
                      1.95104881,  1.50098424,  1.41809404,  1.55529754,  1.03148226,
                      1.50383347,  1.40132244,  1.34442269,  1.52167429,  1.60169425,
                      1.76673304,  1.28942308,  1.44642524,  1.21864227,  1.4202547 ,
                      1.87041141,  2.03483089,  1.71535677,  2.28305006,  1.3655182 ,
                      2.11782833,  1.45678943,  1.5225014 ,  1.67539949,  1.34124091,
                      1.18971384,  1.39146565,  1.64661356,  1.73423454,  1.30871198,
                      1.72766751,  2.09804599,  1.40223231,  2.11199304,  1.1127785 ,
                      1.58594861,  1.54374121,  1.47768618,  1.59315934,  1.42042477,
                      1.93521917,  1.33224972,  1.51089575,  1.44978985,  1.52745566])
        _check_moments(Thetas, ktm, ktv, axis=1)
        
        Sigma_e = np.vstack(self.reg.trace.Sigma_e)
        known_mean = 0.69545342
        known_var = 0.01810897
        check_moments(Sigma_e, known_mean, known_var, axis=None)
        
        Sigma_u = np.vstack(self.reg.trace.Sigma_u)
        known_mean = 1.531941662
        known_var = 1.55432487
        _check_moments(Sigma_u, known_mean, known_var, axis=None)
        
        SAC_Upper = np.vstack(self.reg.trace.SAC_Upper)
        known_mean = 0.04156862
        known_var = 0.10
        _check_moments(SAC_Upper, known_mean, known_var, axis=None)
        
        SAC_Lower = np.vstack(self.reg.trace.SAC_Lower)
        known_mean = -0.01843137
        known_var = 0.004977931
        _check_moments(SAC_Lower, known_mean, known_var, axis=None)

if __name__ == '__main__':
    unittest.main()
