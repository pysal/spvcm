from __future__ import division

import numpy as np
import copy

from ...both_levels.generic import Base_Generic
from ... import verify
from ...utils import se_covariance, ind_covariance


SAMPLERS = ['Alphas', 'Betas', 'Sigma2', 'Tau2', 'Rho']

class Base_Lower_SE(Base_Generic):
    """
    The class that actually ends up setting up the Generic model. Sets configs,
    data, truncation, and initial parameters, and then attempts to apply the
    sample function n_samples times to the state. 
    """
    def __init__(self, Y, X, W, Delta, n_samples=1000, **_configs):
        n_jobs = _configs.pop('n_jobs', 1)
        super(Base_Lower_SE, self).__init__(Y, X, W, np.eye(Delta.shape[1]), Delta, 
                                      n_samples=0, skip_covariance=True, **_configs)
        self.state.Psi_1 = se_covariance
        self.state.Psi_2 = ind_covariance
        self._setup_covariance()
        original_traced = copy.deepcopy(self.traced_params)
        to_drop = [k for k in original_traced if k not in SAMPLERS]
        self.traced_params = copy.deepcopy(SAMPLERS)
        for param in to_drop:
            for i, _  in enumerate(self.trace.chains):
                del self.trace.chains[i][param]

        if n_samples > 0:
            try:
                self.sample(n_samples, n_jobs=n_jobs)
            except (np.linalg.LinAlgError, ValueError) as e:
                Warn('Encountered the following LinAlgError. '
                     'Model will return for debugging. \n {}'.format(e))
    
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
        if st.LogRho0 is None:
            eigenrange = st.Rho_max - st.Rho_min
            Rho_logprior = np.log(1/eigenrange)
            def LogRho0(value):
                return Rho_logprior
            st.LogRho0 = LogRho0

class Lower_SE(Base_Lower_SE): 
    """
    The class that intercepts & validates input
    """
    def __init__(self, Y, X, W, Z=None, Delta=None, membership=None, 
                 #data options
                 transform ='r', n_samples=1000, verbose=False,
                 **options):
        W,_ = verify.weights(W, None, transform=transform)
        self.W = W
        Wmat = W.sparse

        N,_ = X.shape
        if Delta is not None:
            J = Delta.shape[1]
        elif membership is not None:
            J = len(np.unique(membership))

        Delta, membership = verify.Delta_members(Delta, membership, N, J)

        X = verify.covariates(X)

        self._verbose = verbose
        if Z is not None:
            Z = Delta.dot(Z)
            X = np.hstack((X,Z))
        super(Lower_SE, self).__init__(Y, X, Wmat, Delta, n_samples,
                **options)
