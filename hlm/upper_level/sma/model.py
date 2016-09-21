from __future__ import division

import numpy as np
import copy

from ...both_levels.generic import Base_Generic
from ...both_levels.generic.model import SAMPLERS as generic_params
from ... import verify
from ...utils import sma_covariance
from .sample import sample
from ...trace import Trace



SAMPLERS = ['Alphas', 'Betas', 'Sigma2', 'Tau2', 'Lambda']

class Base_Upper_SMA(Base_Generic):
    _sample = sample
    """
    The class that actually ends up setting up the Generic model. Sets configs,
    data, truncation, and initial parameters, and then attempts to apply the
    sample function n_samples times to the state. 
    """
    def __init__(self, y, X, M, Delta, n_samples=1000, **_configs):
        W = np.eye((Delta.shape[0]))
        super(Base_Upper_SMA, self).__init__(y, X, W, M, Delta, 
                                      n_samples=0, skip_covariance=True, **_configs)
        st = self.state
        self.state.Psi_1 = lambda x, Wmat: np.eye(Wmat.shape[0])
        self.state.Psi_2 = sma_covariance
        self._setup_covariance()
        original_traced = copy.deepcopy(self.traced_params)
        extras = [k for k in original_traced if k not in generic_params]
        self.traced_params = copy.deepcopy(SAMPLERS)
        self.traced_params.extend(extras)
        self.trace = Trace(**{k:[] for k in self.traced_params})
       
        st.Lambda_min, st.Lambda_max = -st.Lambda_max, -st.Lambda_min


        try:
            self.sample(n_samples)
        except (np.linalg.LinAlgError, ValueError) as e:
            warn('Encountered the following LinAlgError. '
                 'Model will return for debugging purposes. \n {}'.format(e))

class Upper_SMA(Base_Upper_SMA): 
    """
    The class that intercepts & validates input
    """
    def __init__(self, y, X, M, Z=None, Delta=None, membership=None, 
                 #data options
                 transform ='r', n_samples=1000, verbose=False,
                 **options):
        M, = verify.weights(M, transform=transform)
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
        super(Upper_SMA, self).__init__(y, X, Mmat, Delta, n_samples,
                **options)
