from __future__ import division

import numpy as np
import copy

from ...both_levels.generic import Base_Generic
from ...both_levels.generic.model import SAMPLERS as generic_parameters
from ... import verify
from ...utils import se_covariance, ind_covariance
from .sample import sample
from warnings import warn


SAMPLERS = ['Alphas', 'Betas', 'Sigma2', 'Tau2', 'Lambda']

class Base_Upper_SE(Base_Generic):
    _sample = sample
    """
    The class that actually ends up setting up the Generic model. Sets configs,
    data, truncation, and initial parameters, and then attempts to apply the
    sample function n_samples times to the state. 
    """
    def __init__(self, Y, X, M, Delta, n_samples=1000, **_configs):
        W = np.eye((Delta.shape[0]))
        super(Base_Upper_SE, self).__init__(Y, X, W, M, Delta, 
                                      n_samples=0, skip_covariance=True, **_configs)
        self.state.Psi_1 = ind_covariance 
        self.state.Psi_2 = se_covariance
        self._setup_covariance()
        original_traced = copy.deepcopy(self.traced_params)
        to_drop = [k for k in original_traced if (k not in SAMPLERS and k in generic_parameters)]
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

class Upper_SE(Base_Upper_SE): 
    """
    The class that intercepts & validates input
    """
    def __init__(self, Y, X, M, Z=None, Delta=None, membership=None, 
                 #data options
                 transform ='r', n_samples=1000, verbose=False,
                 **options):
        _, M = verify.weights(None, M, transform=transform)
        self.M = M
        Mmat = M.sparse

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
        super(Upper_SE, self).__init__(Y, X, Mmat, Delta, n_samples,
                **options)
