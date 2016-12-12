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

from ...abstracts import Sampler_Mixin, Trace, Hashmap
from ... import verify
from ...utils import speigen_range, splogdet


SAMPLERS = ['Alphas', 'Betas', 'Sigma2', 'Tau2', 'Gammas', 'Lambda']

class Base_HSDEM(Sampler_Mixin):
    """
    The class that actually ends up setting up the HSDEM model. Sets configs,
    data, truncation, and initial parameters, and then attempts to apply the
    sample function n_samples times to the state. 
    """
    def __init__(self, Y, X, M, Z, Delta, n_samples=1000, **_configs):
        
        N, p = X.shape
        J = M.shape[0]
        WZ = M.dot(Z)
        Z = np.hstack((Z, WZ))
        _J, q = Z.shape
        self.state = Hashmap(**{'X':X, 'y':Y, 'M':M, 'Z':Z, 'Delta':Delta,
                                'N':N, 'J':J, 'p':p, 'q':q })
        self.traced_params = SAMPLERS
        extras = _configs.pop('extra_tracked_params', None)
        if extras is not None:
            self.traced_params.extend(extra_tracked_params)
        self.trace = Trace(**{k:[] for k in self.traced_params})
        leftovers = self._setup_data(**_configs)
        self._setup_configs(**leftovers)
        self._setup_truncation()
        self._setup_initial_values()

        self.cycles = 0
        self.sample(n_samples)

    def _setup_data(self, **hypers):
        In = np.identity(self.state.N)
        Ij = np.identity(self.state.J)
        ## Prior specifications
        Sigma2_s0 = hypers.pop('Sigma2_s0', .001)
        Sigma2_v0 = hypers.pop('Sigma2_v0', .001)
        Betas_cov0 = hypers.pop('Betas_cov0', np.eye(self.state.p) * 100)
        Betas_mean0 = hypers.pop('Betas_mean0', np.zeros((self.state.p, 1)))
        Tau2_s0 = hypers.pop('Tau2_s0', .001)
        Tau2_v0 = hypers.pop('Tau2_v0', .001)
        Gammas_cov0 = hypers.pop('Gammas_cov0', np.eye(self.state.q) * 100)
        Gammas_mean0 = hypers.pop('Gammas_mean0', np.zeros((self.state.q, 1)))

        Betas_covm = np.dot(np.linalg.inv(Betas_cov0), Betas_mean0)
        Gammas_covm = np.dot(np.linalg.inv(Gammas_cov0), Gammas_mean0)
        Tau2_prod = np.dot(Tau2_s0, Tau2_v0)
        Sigma2_prod = np.dot(Sigma2_s0, Sigma2_v0)
        
        XtX = np.dot(self.state.X.T, self.state.X)
        ZtZ = np.dot(self.state.Z.T, self.state.Z)
        DeltatDelta = np.dot(self.state.Delta.T, self.state.Delta)

        innovations = {k:v for k,v in dict(locals()).items() if k not in ['hypers', 'self']}
        self.state.update(innovations)

        return hypers

    def _setup_configs(self, #would like to make these keyword only using * 
                 #multi-parameter options
                 tuning=0, 
                 #spatial parameter metropolis configurations:
                 lambda_jump=.5, lambda_ar_low=.4, lambda_ar_hi=.6, 
                 lambda_proposal=stats.norm, lambda_adapt_step=1.01,
                 **kw):
        """
        Omnibus function to assign configuration parameters to the correct
        configuration namespace
        """
        self.configs = Hashmap()
        self.configs.Lambda = Hashmap()
        self.configs.Lambda.jump = lambda_jump
        self.configs.Lambda.ar_low = lambda_ar_low
        self.configs.Lambda.ar_hi = lambda_ar_hi
        self.configs.Lambda.proposal = lambda_proposal
        self.configs.Lambda.adapt_step = lambda_adapt_step
        self.configs.Lambda.rejected = 0
        self.configs.Lambda.accepted = 0
        self.configs.Lambda.max_adapt = tuning
        self.configs.Lambda.adapt = tuning > 0
        

    def _setup_truncation(self):
        """
        This computes truncations for the spatial parameter, Lambda. If the weights
        matrix is standardized, this will be (1/min_eigenvalue, 1)
        """
        M_emin, M_emax = speigen_range(self.state.M)
        self.state.Lambda_min = 1./M_emin
        self.state.Lambda_max = 1./M_emax
        self.state.logLambda0 = np.log(1/(M_emax - 1/M_emin))

    def _setup_initial_values(self):
        """
        Set abrbitrary starting values for the Metropolis sampler
        """
        Betas = np.zeros((self.state.p ,1))
        Gammas = np.zeros((self.state.q, 1))
        Alphas = np.zeros((self.state.J, 1))
        Sigma2 = 2
        Tau2 = 2
        Lambda = -1.0 / (self.state.N - 1)
        B = spar.csc_matrix(self.state.Ij - Lambda * self.state.M)
        DeltaAlphas = np.dot(self.state.Delta, Alphas)
        XBetas = spdot(self.state.X, Betas)
        ZGammas = np.dot(self.state.Z, Gammas)
        BZGammas = spdot(B, ZGammas)

        innovations = {k:v for k,v in dict(locals()).items() if k not in ['self']}
        self.state.update(innovations)

    _sample = sample

class HSDEM(Base_HSDEM): 
    """
    The class that intercepts & validates input
    """
    def __init__(self, Y, X, M, Z=None, Delta=None, membership=None, 
                 #data options
                 sparse = True, transform ='r', n_samples=1000, verbose=False,
                 **options):
        M, = verify.weights(M,transform=transform)
        self.M = M

        N,_ = X.shape
        J = M.n
        Mmat = M.sparse

        Delta, membership = verify.Delta_members(Delta, membership, N, J)

        X = verify.covariates(X)

        self._verbose = verbose
        if Z is None:
            Z = np.zeros((J, 1))
        super(HSDEM, self).__init__(Y, X, Mmat, Z, Delta, n_samples,
                **options)
