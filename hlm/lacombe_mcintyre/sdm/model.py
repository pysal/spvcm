from __future__ import division

import types
import numpy as np
import scipy.stats as stats
import scipy.sparse as spar

from numpy import linalg as la
from warnings import warn as Warn
from pysal.spreg.utils import sphstack, spdot
from .sample import sample
from . import verify
from ..utils import speigen_range, splogdet, Namespace as NS


SAMPLERS = ['Alphas', 'Betas', 'Sigma', 'Tau', 'Gamma', 'Rho']

def _keep(k,v, *matches):
    keep = True
    keep &= not isinstance(v, (types.ModuleType, types.FunctionType,
                               types.BuiltinFunctionType, 
                               types.BuiltinMethodType, type))
    keep &= not k.startswith('_')
    keep &= not (k is'self')
    keep &= not (k in matches)
    return keep

class Base_HSDM(object):
    """
    The class that actually ends up setting up the HSDM model. Sets configs,
    data, truncation, and initial parameters, and then attempts to apply the
    sample function n_samples times to the state. 
    """
    def __init__(self, y, X, W, M, Z, Delta, n_samples=1000, **_configs):
    
        self.state = NS()
        self._setup_data()
        self._setup_configs()
        self._setup_truncation()
        self._setup_initial_values()

        self.sample(n_samples)

class HSDM(Base_SDM): 
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
