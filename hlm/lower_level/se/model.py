from __future__ import division

import numpy as np
import copy

from ...both_levels.generic import Base_Generic
from ... import verify
from ...utils import se_covariance


SAMPLERS = ['Alphas', 'Betas', 'Sigma2', 'Tau2', 'Rho']

class Base_Lower_SE(Base_Generic):
    """
    The class that actually ends up setting up the Generic model. Sets configs,
    data, truncation, and initial parameters, and then attempts to apply the
    sample function n_samples times to the state. 
    """
    def __init__(self, y, X, W, Delta, n_samples=1000, **_configs):
        super(Base_Lower_SE, self).__init__(y, X, W, np.eye(Delta.shape[1]), Delta, 
                                      n_samples=0, skip_covariance=True, **_configs)
        self.state.Psi_1 = se_covariance
        self.state.Psi_2 = lambda x, Wmat: np.eye(Wmat.shape[0])
        self._setup_covariance()
        original_traced = copy.deepcopy(self.traced_params)
        to_drop = [k for k in original_traced if k not in SAMPLERS]
        self.traced_params = SAMPLERS
        for param in to_drop:
            del self.trace[param]

        self.sample(n_samples)

class Lower_SE(Base_Lower_SE): 
    """
    The class that intercepts & validates input
    """
    def __init__(self, y, X, W, Z=None, Delta=None, membership=None, 
                 #data options
                 transform ='r', n_samples=1000, verbose=False,
                 **options):
        W, = verify.weights(W, transform=transform)
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
        super(Lower_SE, self).__init__(y, X, Wmat, Delta, n_samples,
                **options)
