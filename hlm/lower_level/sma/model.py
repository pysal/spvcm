from __future__ import division

import numpy as np
import copy

from ...both_levels.generic import Base_Generic
from ... import verify
from ...utils import sma_covariance, ind_covariance


SAMPLERS = ['Alphas', 'Betas', 'Sigma2', 'Tau2', 'Rho']

class Base_Lower_SMA(Base_Generic):
    """
    The class that actually ends up setting up the Generic model. Sets configs,
    data, truncation, and initial parameters, and then attempts to apply the
    sample function n_samples times to the state.
    """
    def __init__(self, Y, X, W, Delta, n_samples=1000, n_jobs=1,
                 extra_traced_params = None,
                 priors=None,
                 starting_values=None,
                 configs=None,
                 truncation=None):
        M = np.eye(Delta.shape[1])
        super(Base_Lower_SMA, self).__init__(Y, X, W, M, Delta,
                                            n_samples=0, n_jobs=n_jobs,
                                            extra_traced_params=extra_traced_params,
                                            priors=priors,
                                            configs=configs,
                                            starting_values=starting_values,
                                            truncation=truncation)

        original_traced = copy.deepcopy(self.traced_params)
        to_drop = [k for k in original_traced if k not in SAMPLERS]
        self.traced_params = copy.deepcopy(SAMPLERS)
        for param in to_drop:
            for i, _  in enumerate(self.trace.chains):
                del self.trace.chains[i][param]

        self.state.Psi_1 = sma_covariance
        self.state.Psi_2 = ind_covariance

        if n_samples > 0:
            try:
                self.sample(n_samples, n_jobs=n_jobs)
            except (np.linalg.LinAlgError, ValueError) as e:
                Warn('Encountered the following LinAlgError. '
                     'Model will return for debugging. \n {}'.format(e))

class Lower_SMA(Base_Lower_SMA):
    """
    The class that intercepts & validates input
    """
    def __init__(self, Y, X, W, Z=None, Delta=None, membership=None,
                 #data options
                 transform ='r', verbose=False,
                 n_samples=1000, n_jobs=1,
                 extra_traced_params = None,
                 priors=None,
                 configs=None,
                 starting_values=None,
                 truncation=None,
                 center=True,
                 scale=False):
        W,_ = verify.weights(W, None, transform=transform)
        self.W = W
        Wmat = W.sparse

        N,_ = X.shape
        if Delta is not None:
            J = Delta.shape[1]
        elif membership is not None:
            J = len(np.unique(membership))

        Delta, membership = verify.Delta_members(Delta, membership, N, J)
        if Z is not None:
            Z = Delta.dot(Z)
            X = np.hstack((X,Z))
        if center:
            X = verify.center(X)
        if scale:
            X = verify.scale(X)

        X = verify.covariates(X)

        self._verbose = verbose

        super(Lower_SMA, self).__init__(Y, X, Wmat, Delta,
                                       n_samples=n_samples,
                                       n_jobs = n_jobs,
                                       extra_traced_params=extra_traced_params,
                                       priors=priors,
                                       configs=configs,
                                       starting_values=starting_values,
                                       truncation=truncation)
