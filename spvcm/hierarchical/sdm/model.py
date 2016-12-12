from __future__ import division

import numpy as np
import scipy.stats as stats
import scipy.sparse as spar
import copy

from numpy import linalg as la
from warnings import warn as Warn
from pysal.spreg.utils import spdot
from .sample import sample

from ...abstracts import Sampler_Mixin, Trace, Hashmap
from ... import verify
from ...utils import speigen_range 


SAMPLERS = ['Alphas', 'Betas', 'Sigma2', 'Tau2', 'Gammas', 'Rho']

class Base_HSDM(Sampler_Mixin):
    """
    The class that actually ends up setting up the HSDM model. Sets configs,
    data, truncation, and initial parameters, and then attempts to apply the
    sample function n_samples times to the state. 
    """
    def __init__(self, Y, X, M, Z, Delta, n_samples=1000, **_configs):
        
        N, p = X.shape
        WZ = M.dot(Z)
        Z = np.hstack((Z, WZ))
        J, q = Z.shape
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
        Betas_cov0 = hypers.pop('Betas_cov0', np.eye(self.state.p) * .001)
        Betas_mean0 = hypers.pop('Betas_mean0', np.zeros((self.state.p, 1)))
        Tau2_s0 = hypers.pop('Tau2_s0', .001)
        Tau2_v0 = hypers.pop('Tau2_v0', .001)
        Gammas_cov0 = hypers.pop('Gammas_cov0', np.eye(self.state.q) * .001)
        Gammas_mean0 = hypers.pop('Gammas_mean0', np.zeros((self.state.q, 1)))

        Betas_covm = np.dot(Betas_cov0, Betas_mean0)
        Gammas_covm = np.dot(Gammas_cov0, Gammas_mean0)
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
                 truncate='eigs', tuning=0, 
                 #spatial parameter grid sample configurations:
                 rho_jump=.5, rho_ar_low=.4, rho_ar_hi=.6, 
                 rho_proposal=stats.norm, rho_adapt_step=1.01,
                 **kw):
        """
        Omnibus function to assign configuration parameters to the correct
        configuration namespace
        """
        self.configs = Hashmap()
        self.configs.truncate = truncate
        self.configs.Rho = Hashmap()
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

    def _setup_truncation(self):
        """
        This computes truncations for the spatial parameters. 

        If configs.truncate is set to 'eigs', computes the eigenrange of the two
        spatial weights matrices using speigen_range

        If configs.truncate is set to 'stable', sets the truncation to -1,1
        
        If a tuple is passed to truncate, then this will truncate the
        distribution according to this tuple
        """
        M_emin, M_emax = speigen_range(self.state.M)
        self.state.Rho_min = 1./M_emin
        self.state.Rho_max = 1./M_emax

    def _setup_initial_values(self):
        """
        Set abrbitrary starting values for the Metropolis sampler
        """
        Betas = np.zeros((self.state.p ,1))
        Gammas = np.zeros((self.state.q, 1))
        Alphas = np.zeros((self.state.J, 1))
        Sigma2 = 2
        Tau2 = 2
        Rho = -1.0 / (self.state.J - 1)
        B = spar.csc_matrix(self.state.Ij - Rho * self.state.M)
        DeltaAlphas = np.dot(self.state.Delta, Alphas)
        XBetas = np.dot(self.state.X, Betas)
        ZGammas = np.dot(self.state.Z, Gammas)
        BZGammas = spdot(B, ZGammas)
        
        innovations = {k:v for k,v in dict(locals()).items() if k not in ['self']}
        self.state.update(innovations)

    _sample = sample

class HSDM(Base_HSDM): 
    """
    The class that intercepts & validates input
    """
    def __init__(self, Y, X, M, Z=None, Delta=None, membership=None, 
                 #data options
                 sparse = True, transform ='r', n_samples=1000, verbose=False,
                 **options):
        M, = verify.weights(M, transform=transform)
        self.M = M

        N,_ = X.shape
        J = M.n
        Mmat = M.sparse

        Delta, membership = verify.Delta_members(Delta, membership, N, J)

        X = verify.covariates(X)

        self._verbose = verbose
        if Z is None:
            Z = np.zeros((J, 1))
        super(HSDM, self).__init__(Y, X, Mmat, Z, Delta, n_samples,
                **options)
