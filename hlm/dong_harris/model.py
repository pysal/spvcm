from __future__ import division

import numpy as np
from numpy import linalg as la
from warnings import warn as Warn
from six import iteritems as diter

from pysal.spreg.utils import sphstack, spdot

from . import verify
import types

from ..abstracts import AbstractSampler, Gibbs
from ..utils import grid_det, Namespace as NS
try:
    from ..utils import theano_grid_det
except ImportError:
    pass

from . import samplers as dh_samplers
_HSAR_SAMPLERS = [dh_samplers.__dict__[S] for S in dh_samplers.__all__]

def _keep(k,v, *matches):
    keep = True
    keep &= not isinstance(v, (types.ModuleType, types.FunctionType,
                               types.BuiltinFunctionType, 
                               types.BuiltinMethodType, type))
    keep &= not k.startswith('_')
    keep &= not (k is'self')
    keep &= not (k in matches)
    return keep

class Base_HSAR(Gibbs):
    def __init__(self, y, X, W, M, Z, Delta, SAC_Upper_grid, SAC_Lower_grid, 
                 cycles=1000, steps=0, **tuning):
        if Z is not None:
            X = sphstack(X, spdot(Delta, Z))
        del Z

        # dimensions
        N, p = X.shape
        J = M.shape[0]
        self._state = NS(**{k:v for k,v in diter(locals()) if _keep(k,v)}) 

        initial_state, leftovers = self._setup_data(**tuning)
        self._state.update(initial_state)
        
        hypers = {k:self._state[k] for k in ['M0', 'T0', 'a0', 'b0', 'c0', 'd0']}
        self.hypers = NS(**hypers)
        
        samplers, leftovers = self._setup_samplers(**leftovers)
        initial_values = {k.__class__.__name__:k.initial for k in samplers}
        self._state.update(initial_values)
        super(Base_HSAR, self).__init__(*samplers, state=self._state)
        
        self.sample(cycles=cycles, steps=steps)

    def _setup_data(self, **tuning):
        """
        This sets up the same example problem as in the Dong & Harris HSAR code. 
        """

        In = np.identity(self._state.N)
        Ij = np.identity(self._state.J)
        ##Prior specs
        M0 = tuning.pop('M0', np.zeros(self._state.p))
        T0 = tuning.pop('T0', np.identity(self._state.p) * 100)
        a0 = tuning.pop('a0', .01)
        b0 = tuning.pop('b0', .01)
        c0 = tuning.pop('c0', .01)
        d0 = tuning.pop('d0', .01)

        ##fixed matrix manipulations for MCMC loops
        XtX = spdot(self._state.X.T, self._state.X)
        invT0 = la.inv(T0)
        T0M0 = spdot(invT0, M0)

        ##unchanged posterior conditionals for sigma_e, sigma_u
        ce = self._state.N/2. + c0
        au = self._state.J/2. + a0

        ##set up griddy gibbs
        
        #invariants in rho sampling
        beta0, resids, rank, svs = la.lstsq(self._state.X, self._state.y)
        e0 = self._state.y - spdot(self._state.X, beta0)
        e0e0 = spdot(e0.T, e0)

        Wy = spdot(self._state.W, self._state.y)
        betad, resids, rank, svs = la.lstsq(self._state.X, Wy)
        ed = Wy - spdot(self._state.X, betad)
        eded = spdot(ed.T, ed)
        e0ed = spdot(e0.T, ed)

        rval = {k:v for k,v in diter(dict(locals())) if k is not 'tuning'}
        return rval, tuning

    def _setup_samplers(self, **start):
        samplers = []
        for S in _HSAR_SAMPLERS:
            guess = start.pop(S.__name__, None)
            samplers.append(S(state=self, initial=guess))
        return samplers, start

class HSAR(Base_HSAR):
    def __init__(self, y, X, W, M, 
                 Z=None, Delta=None, membership=None,
                 err_grid=None, err_gridfile='', sar_grid=None, sar_gridfile='', 
                 sparse=True, transform='r', cycles=1000, steps=0,
                 verbose=False, **tuning):
        """
        The Dong-Harris multilevel HSAR model

        Parameters
        ----------
        y               response
        X               lower covariates
        W               lower weights
        M               upper weights
        Z               optional upper covariates
        Delta           aggregation matrix classifing W into M
        membership      vector containing classification of W into M
        err_grid        tuple containing (min,max,step) for grid to sample lambda
                        or  
                        array containing grid of lambda values to use
        sar_grid        tuple containing (min,max,step) for grid to sample rho
                        or 
                        array containing grid of rho values to use
        err_gridfile    string or file specifying a stored grid of log
                        determinants and parameter values to use for lambda
        sar_gridfile    string or file specifying a stored grid of log
                        determinants and parameter values to use for rho
        transform       weights transformation 
        """
        # Weights & Projections
        W,M = verify.weights(W,M,transform)
        self.W = W
        self.M = M
        N,J = W.n, M.n 
        _N, _ = X.shape
        try:
            assert _N == N
        except AssertionError:
            raise UserWarning('Number of lower-level observations does not match between X ({}) and W ({})'.format(_N, N))

        if sparse:
            Wmat = W.sparse
            Mmat = M.sparse
        else:
            Wmat = W.full()[0]
            Mmat = M.full()[0]
        
        Delta, membership = verify.Delta_members(Delta, membership, N, J)
        
        # Data
        X = verify.covariates(X, W)
        
        # Gridded SAR/ER
        if (err_grid is None) & (err_gridfile is ''):
            err_grid = (-.99,.99,.01)
        if (sar_grid is None) & (sar_gridfile is ''):
            sar_grid = (-.99,.99,.01)
        err_prom = verify.parameters(err_grid, err_gridfile, Mmat) #call to compute
        sar_prom = verify.parameters(sar_grid, sar_gridfile, Wmat) #call to compute

        self._verbose = verbose
        super(HSAR, self).__init__(y, X, Wmat, Mmat, Z, 
                                   Delta, err_prom(), sar_prom(), 
                                   cycles=cycles, steps=steps, **tuning)

if __name__ == '__main__':
    import pandas as pd

    data = pd.read_csv('./test.csv')
    y = data[['y']].values
    X = data[['X']].values
    
    W_low = ps.open('w_lower.mtx').read()
    W_low.transform = 'r'
    W_up = ps.open('w_upper.mtx').read()
    W_up.transform = 'r'
    
    membership = data[['county']].values

    Z = np.ones(W_low.n).reshape((W_upper.n, 1))
    
    test = HSAR(y, X, W_low, W_up, Z=Z, membership=membership)
