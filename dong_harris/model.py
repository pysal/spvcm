from __future__ import division

import numpy as np
from numpy import linalg as la
from warnings import warn as Warn

from pysal.spreg.utils import sphstack, spdot

from . import verify

from ..samplers import AbstractSampler, Gibbs
from ..utils import grid_det
try:
    from ..utils import theano_grid_det
except ImportError:
    pass

from . import samplers as dh_samplers
_HSAR_SAMPLERS = [dh_samplers.__dict__[S] for S in dh_samplers.__all__]

class Base_HSAR(Gibbs):
    def __init__(self, y, X, Wmat, Mmat, Z, Delta, SAC_upper, SAC_lower, 
                 cycles=1000, steps=0, **tuning):
        self._state = {} 
        
        # dimensions
        self.N, self.pq = X.shape
        self.pq -= 2 #upper and lower constants removed
        self.J, self.q = Z.shape
        self.q -= 1 #lower constant removed
        self.p = self.pq - self.q

        # grids
        self._SAC_upper = SAC_upper
        self._SAC_lower = SAC_lower

        # weights
        self._Wmat = Wmat
        self._Mmat = Mmat

        initial_state, leftovers = self._setup_data(**tuning)
        self._state.update(initial_state)
        hypers = {k:self._state[k] for k in ['M0', 'T0', 'a0', 'b0', 'c0', 'd0']}
        self.__dict__.update(hypers) #want these at top level for users to see
        
        self.samplers = self._setup_samplers(**leftovers)
        self.trace = {k:[self._state[k]] for k in _HSAR_SAMPLERS}
        
        self.sample(cycles=cycles, steps=steps)

    def _setup_date(self, **tuning):
        """
        This sets up the same example problem as in the Dong & Harris HSAR code. 
        """

        In = np.identity(self.N)
        Ij = np.identity(self.J)

        ##Prior specs
        M0 = tuning.pop('M0', np.zeros(self.p))
        T0 = tuning.pop('T0', np.identity(self.p) * 100)
        a0 = tuning.pop('a0', .01)
        b0 = tuning.pop('b0', .01)
        c0 = tuning.pop('c0', .01)
        d0 = tuning.pop('d0', .01)

        ##fixed matrix manipulations for MCMC loops
        XtX = spdot(X.T, X)
        invT0 = la.inv(T0)
        T0M0 = spdot(invT0, M0)

        ##unchanged posterior conditionals for sigma_e, sigma_u
        ce = N/2. + c0
        au = J/2. + a0

        ##set up griddy gibbs
        
        #invariants in rho sampling
        beta0, resids, rank, svs = la.lstsq(X, y)
        e0 = y - spdot(X, beta0)
        e0e0 = spdot(e0.T, e0)

        Wy = spdot(W, y)
        betad, resids, rank, svs = la.lstsq(X, Wy)
        ed = Wy - spdot(X, betad)
        eded = spdot(ed.T, ed)
        e0ed = spdot(e0.T, ed)

        return locals(), tuning

    def _setup_samplers(self, **start):
        samplers = []
        for sampler in _HSAR_SAMPLERS:
            guess = start.pop(S.__name__, None)
            samplers.append(S(state=self._state, initial=guess))
        return samplers, start

class HSAR(Base_HSAR):
    def __init__(self, y, X, W, M, 
                 Z=None, Delta=None, membership=None,
                 err_range=(), err_gridfile='', sar_range=(), sar_gridfile='', 
                 sparse=True, transform='r', cycles=1000, steps=0, **tuning):
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
        sar_grid        tuple containing (min,max,step) for grid to sample rho
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
            raise AssertionError('Number of lower-level observations does not match between X ({}) and W ({})'.format(_N, N))

        self.J = J
        
        if sparse:
            Wmat = W.sparse
            Mmat = M.sparse
        else:
            Wmat = W.full()[0]
            Mmat = M.full()[0]
        
        Delta, membership = verify.Delta_members(Delta, membership, N, J)
        
        # Data
        X, Z = verify.covariates(X, Z, W, M)
        X = sphstack(X,  Z)
        
        # Gridded SAR/ER
        err_prom = verify.parameters(err_grid, err_gridfile, Mmat) #call to compute
        sar_prom = verify.parameters(sar_grid, sar_gridfile, Wmat) #call to compute

        super(self, HSAR).__init__(y, X, Wmat, Mmat, Z, 
                                   Delta, err_prom(), sar_prom(), **tuning)

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

    Z = np.ones(W_low.n).reshape((W_low.n, 1))
    
    test = HSAR(y, X, W_low, W_up, Z=None, membership=membership)
