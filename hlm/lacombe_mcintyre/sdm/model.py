from __future__ import division

import types
import numpy as np
import scipy.stats as stats
import scipy.sparse as spar

from numpy import linalg as la
from warnings import warn as Warn
from pysal.spreg.utils import sphstack, spdot
from .sample import sample
from ... import verify
from ...utils import speigen_range, splogdet, Namespace as NS


SAMPLERS = ['Alphas', 'Betas', 'Sigma', 'Tau', 'Gamma', 'Rho']

class Base_HSDM(object):
    """
    The class that actually ends up setting up the HSDM model. Sets configs,
    data, truncation, and initial parameters, and then attempts to apply the
    sample function n_samples times to the state. 
    """
    def __init__(self, y, X, W, M, Z, Delta, n_samples=1000, **_configs):
        
        N, p = X.shape
        J = M.shape[0]
        _J, q = Z.shape
        self.state = NS(**{'X':X, 'y':y, 'W':W, 'M':M, 'Z':Z, 'Delta':Delta,
                           'N':N, 'J':J, 'p':p, 'q':q })
        self.trace = NS
        self.traced_params = SAMPLERS
        extras = _configs.pop('extra_tracked_params', None)
        if extras is not None:
            self.traced_params.extend(extra_tracked_params)
        initial_state, leftovers = self._setup_data(**_configs)
        self._setup_configs()
        self._setup_truncation()
        self._setup_initial_values()
        self.sample(n_samples)

    def _setup_data(self, **tuning):
        In = np.identity(self.state.N)
        Ij = np.identity(self.state.J)
        ## Prior specifications
        Sigma2_s0 = tuning.pop('Sigma2_s0', .001)
        Sigma2_v0 = tuning.pop('Sigma2_v0', .001)
        Betas_cov0 = tuning.pop('Betas_cov0', np.eye(self.state.p) * .001)
        Betas_mean0 = tuning.pop('Betas_mean0', np.zeros((p, 1)))
        Tau2_s0 = tuning.pop('Tau2_s0', .001)
        Tau2_v0 = tuning.pop('Tau2_v0', .001)
        Gammas_cov0 = tuning.pop('Gammas_cov0', np.zeros((q, 1 )))
        Gammas_mean0 = tuning.pop('Gammas_mean0', np.eye(q) * .001)

        Betas_covm = np.dot(Betas_cov0, Betas_mean0)
        Gammas_covm = np.dot(Gammas_cov0, Gammas_mean0)
        Tau2_prod = np.dot(Tau2_s0, Tau2_v0)
        Sigma2_prod = np.dot(Sigma2_s0, Sigma2_v0)

    def _setup_configs():
        pass

    def _setup_truncation(self):
        """
        This computes truncations for the spatial parameters. 

        If configs.truncate is set to 'eigs', computes the eigenrange of the two
        spatial weights matrices using speigen_range

        If configs.truncate is set to 'stable', sets the truncation to -1,1

        If configs.truncate is a tuple of values, this attempts to interpret
        them as separate assignments for Rho and Lambda truncations first:
        (1/Rho_min, 1/Rho_max, 1/Lambda_min, 1/Lambda_max)
        and then as joint assignments such that:
        (1/Rho_min = 1/Lambda_min = 1/Joint_min,
         1/Rho_max = 1/Lambda_max = 1/Joint_max,)
        """
        state = self.state
        if self.configs.truncate == 'eigs':
            M_emin, M_emax = speigen_range(state.M)
        elif self.configs.truncate == 'stable':
            W_emax = W_emax = 1
        elif isinstance(self.configs.truncate, tuple):
            try:
                W_emin, W_emax, M_emin, M_emax = self.configs.truncate
            except ValueError:
                W_emin, W_emax = self.configs.truncate
                M_emin, M_emax = W_emin, W_emax
        else:
            raise Exception('Truncation parameter was not understood.')
        state.Rho_min = 1./M_emin
        state.Rho_max = 1./M_emax

    def _setup_initial_values():
        pass
    

class HSDM(Base_HSDM): 
    """
    The class that intercepts & validates input
    """
    def __init__(self, y, X, W, M, Z=None, Delta=None, membership=None, 
                 #data options
                 sparse = True, transform ='r', n_samples=1000, verbose=False,
                 **options):
        W, M = verify.weights(W, M, transform)
        self.W = W
        self.M = M
        N, J = W.n, M.n
        _N, _ = X.shape
        try:
            assert _N == N
        except AssertionError:
            raise UserWarning('Number of lower-level observations does not match'
                    ' between X ({}) and W ({})'.format(_N, N))
        Wmat = W.sparse
        Mmat = M.sparse

        Delta, membership = verify.Delta_members(Delta, membership, N, J)

        X = verify.covariates(X, W)

        self._verbose = verbose
        super(HSDM, self).__init__(y, X, Wmat, Mmat, Z, Delta, n_samples,
                **options)
        pass
