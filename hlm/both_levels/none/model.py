from __future__ import division

import types
import numpy as np
import scipy.stats as stats
import scipy.sparse as spar
import copy

from numpy import linalg as la
from warnings import warn as Warn
from pysal.spreg.utils import sphstack, spdot
from .sample import sample
from ...abstracts import Sampler_Mixin
from ... import verify
from ...utils import speigen_range, splogdet
from ...trace import Trace


SAMPLERS = ['Alphas', 'Betas', 'Sigma2', 'Tau2']

class Base_MVCM(Sampler_Mixin):
    """
    The class that actually ends up setting up the MVCM model. Sets configs,
    data, truncation, and initial parameters, and then attempts to apply the
    sample function n_samples times to the state. 
    """
    def __init__(self, y, X, Delta, n_samples=1000, **_configs):
        
        N, p = X.shape
        _N, J = Delta.shape
        self.state = Trace(**{'X':X, 'y':y, 'Delta':Delta,
                           'N':N, 'J':J, 'p':p })
        self.trace = Trace()
        self.traced_params = SAMPLERS
        extras = _configs.pop('extra_tracked_params', None)
        if extras is not None:
            self.traced_params.extend(extra_tracked_params)
        self.trace.update({k:[] for k in self.traced_params})
        leftovers = self._setup_data(**_configs)
        self._setup_configs(**leftovers)
        self._setup_initial_values()
        self._setup_covariance()
        
        self.cycles = 0
        

        try:
            self.sample(n_samples)
        except (np.linalg.LinAlgError, ValueError) as e:
            Warn('Encountered the following LinAlgError. '
                 'Model will return for debugging. \n {}'.format(e))


    def _setup_data(self, **hypers):
        In = np.identity(self.state.N)
        Ij = np.identity(self.state.J)
        ## Prior specifications
        Sigma2_a0 = hypers.pop('Sigma2_a0', .001)
        Sigma2_b0 = hypers.pop('Sigma2_b0', .001)
        Betas_cov0 = hypers.pop('Betas_cov0', np.eye(self.state.p) * 100)
        Betas_mean0 = hypers.pop('Betas_mean0', np.zeros((self.state.p, 1)))
        Tau2_a0 = hypers.pop('Tau2_a0', .001)
        Tau2_b0 = hypers.pop('Tau2_b0', .001)


        XtX = np.dot(self.state.X.T, self.state.X)
        DeltatDelta = np.dot(self.state.Delta.T, self.state.Delta)
        
        innovations = {k:v for k,v in dict(locals()).items() if k not in ['hypers', 'self']}
        self.state.update(innovations)

        return hypers
    
    def _finalize_invariants(self):
        """
        This computes derived properties of hyperparameters that do not change 
        over iterations. This is called one time before sampling. 
        """
        st = self.state
        st.Betas_cov0i = np.linalg.inv(st.Betas_cov0)
        st.Betas_covm = np.dot(st.Betas_cov0, st.Betas_mean0)
        st.Sigma2_an = self.state.N / 2 + st.Sigma2_a0
        st.Tau2_an = self.state.J / 2 + st.Tau2_a0

    def _setup_configs(self, #would like to make these keyword only using * 
                 #multi-parameter options
                 truncate='eigs', tuning=0, 
                 #spatial parameter grid sample configurations:
                 rho_jump=.5, rho_ar_low=.4, rho_ar_hi=.6, 
                 rho_proposal=stats.norm, rho_adapt_step=1.01,
                 #spatial parameter grid sample configurations:
                 lambda_jump=.5, lambda_ar_low=.4, lambda_ar_hi=.6, 
                 lambda_proposal=stats.norm, lambda_adapt_step=1.01,
                 **kw):
        """
        Omnibus function to assign configuration parameters to the correct
        configuration namespace
        """
        self.configs = Trace()
        self.configs.Rho = Trace()
        self.configs.Rho.jump = rho_jump
        self.configs.Rho.ar_low = rho_ar_low
        self.configs.Rho.ar_hi = rho_ar_hi
        self.configs.Rho.proposal = rho_proposal
        self.configs.Rho.adapt_step = rho_adapt_step
        self.configs.Rho.rejected = 0
        self.configs.Rho.accepted = 0
        self.configs.Rho.max_adapt = tuning 
        if tuning > 0:
            self.configs.Rho.adapt = True
        else:
            self.configs.Rho.adapt = False
        self.configs.Lambda = Trace()
        self.configs.Lambda.jump = lambda_jump
        self.configs.Lambda.ar_low = lambda_ar_low
        self.configs.Lambda.ar_hi = lambda_ar_hi
        self.configs.Lambda.proposal = lambda_proposal
        self.configs.Lambda.adapt_step = lambda_adapt_step
        self.configs.Lambda.rejected = 0
        self.configs.Lambda.accepted = 0
        self.configs.Lambda.max_adapt = tuning 
        if tuning > 0:
            self.configs.Lambda.adapt = True
        else:
            self.configs.Lambda.adapt = False

    def _setup_initial_values(self):
        """
        Set abrbitrary starting values for the Metropolis sampler
        """
        Betas = np.zeros((self.state.p ,1))
        Alphas = np.zeros((self.state.J, 1))
        Sigma2 = 4
        Tau2 = 4
        DeltaAlphas = np.dot(self.state.Delta, Alphas)
        XBetas = np.dot(self.state.X, Betas)
        
        innovations = {k:v for k,v in dict(locals()).items() if k not in ['self']}
        self.state.update(innovations)

    def _setup_covariance(self):
        """
        Set up covariance, depending on model form. If this is SMA, the
        utils.sma_covariance function will be used. If this is SAR-error, the
        utils.ser_covariance function will be used.
        """
        st = self.state
        st.Psi_1 = lambda par, Wobj: np.eye(Wobj.shape[0])
        st.Psi_2 = lambda par, Wobj: np.eye(Wobj.shape[0])
        
        st.PsiRho = st.In
        st.PsiLambda = st.Ij

        st.PsiSigma2 = st.In * st.Sigma2
        st.PsiSigma2i = la.inv(st.PsiSigma2)
        st.PsiTau2 = st.Ij * st.Tau2
        st.PsiTau2i = la.inv(st.PsiTau2)
        
        st.PsiRhoi = la.inv(st.PsiRho)
        st.PsiLambdai = la.inv(st.PsiLambda)
    
    _sample = sample

class MVCM(Base_MVCM): 
    """
    The class that intercepts & validates input
    """
    def __init__(self, y, X, Z=None, Delta=None, membership=None, 
                 #data options
                 sparse = True, transform ='r', n_samples=1000, verbose=False,
                 **options):

        N, _ = X.shape
        if Delta is not None:
            _,J = Delta.shape
        elif membership is not None:
            J = len(np.unique(membership))

        Delta, membership = verify.Delta_members(Delta, membership, N, J)

        X = verify.covariates(X)

        self._verbose = verbose
        if Z is not None:
            Z = Delta.dot(Z)
            X = np.hstack((X,Z))
        super(MVCM, self).__init__(y, X, Delta, n_samples,
                **options)
