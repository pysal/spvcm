from __future__ import division

import numpy as np
import scipy.stats as stats
import scipy.sparse as spar
from numpy import linalg as la
from warnings import warn as Warn
from pysal.spreg.utils import sphstack, spdot
from .sample import sample

from . import verify
import types

from ..utils import speigen_range, splogdet, Namespace as NS

SAMPLERS = ['Betas', 'Thetas', 'Sigma2_e', 'Sigma2_u', 'Rho', 'Lambda']


def _keep(k,v, *matches):
    keep = True
    keep &= not isinstance(v, (types.ModuleType, types.FunctionType,
                               types.BuiltinFunctionType, 
                               types.BuiltinMethodType, type))
    keep &= not k.startswith('_')
    keep &= not (k is'self')
    keep &= not (k in matches)
    return keep

class Base_HSAR(object):
    def __init__(self, y, X, W, M, Z, Delta, 
                 n_samples=1000, **_configs):
        if Z is not None:
            X = sphstack(X, spdot(Delta, Z))
        del Z

        # dimensions
        N, p = X.shape
        J = M.shape[0]
        self.state = NS(**{k:v for k,v in dict(locals()).items() if _keep(k,v)}) 
        self.trace = NS()
        self.traced_params = SAMPLERS
        extras = _configs.pop('extra_tracked_params', None)
        if extras is not None:
            self.traced_params.extend(extra_tracked_params)
        self.trace.update({k:[] for k in self.traced_params})

        initial_state, leftovers = self._setup_data(**_configs)
        self.state.update(initial_state)
        self._setup_configs(**_configs)
        self._setup_initial_values(**_configs)
        self._setup_truncation()
        if self.configs.Rho.sample_method.startswith('grid'):
            self._setup_grid(self.configs.Rho, self.state.Rho_min, 
                             self.state.Rho_max, self.state.W, self.state.In)
        if self.configs.Lambda.sample_method.startswith('grid'):
            self._setup_grid(self.configs.Lambda, self.state.Lambda_min,
                             self.state.Lambda_max, self.state.M, self.state.Ij)

        self.state._n_iterations = 0
        self.cycles = 0
        
        self.sample(n_samples)

    def _setup_configs(self, 
                 #multi-parameter options
                 effects_method='cho', spatial_method='grid',
                 truncate='eigs', tuning=0, 
                 #spatial parameter grid sample configurations:
                 lambda_grid=None, rho_grid=None, 
                 #spatial parameter metropolis sample configurations
                 rho_jump=.5, rho_ar_low=.4, rho_ar_hi=.6, 
                 rho_proposal=stats.norm, rho_adapt_step=1.001,
                 lambda_jump=.5, lambda_ar_low=.4, lambda_ar_hi=.6, 
                 lambda_proposal=stats.norm, lambda_adapt_step=1.001,
                 #analytical parameter options:
                 betas_overwrite_covariance=True,
                 thetas_overwrite_covariance=True):
        """
        Omnibus function to assign configuration parameters to the correct
        configuration namespace
        """
        to_apply = locals()
        self.configs = NS()
        for sampler in SAMPLERS:
            self.configs.update({sampler:NS()})
            confs = self.configs[sampler]
            apply_to_this = {k:v for k,v in to_apply.items() 
                             if k.startswith(sampler.lower())}
            apply_to_this = {k.lstrip(sampler.lower() + '_'):v 
                             for k,v in apply_to_this.items()}
            confs.update(apply_to_this)
        self.configs.Rho.tuning = tuning
        self.configs.Lambda.tuning = tuning
        self.configs.Rho.sample_method = spatial_method
        self.configs.Lambda.sample_method = spatial_method
        
        self.configs.Betas.sample_method = effects_method
        self.configs.Thetas.sample_method = effects_method
        self.configs.truncate = truncate

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
            W_emin, W_emax = speigen_range(state.W)
            M_emin, M_emax = speigen_range(state.M)
        elif self.configs.truncate == 'stable':
            W_emin = M_emin = -1
            W_emax = W_emax = 1
        elif isinstance(self.configs.truncate, tuple):
            try:
                W_emin, W_emax, M_emin, M_emax = truncate
            except ValueError:
                W_emin, W_emax = truncate
                M_emin, M_emax = W_emin, W_emax
        else:
            raise Exception('Truncation parameter was not understood.')
        state.Rho_min = 1./W_emin
        state.Rho_max = 1./W_emax
        state.Lambda_min = 1./M_emin
        state.Lambda_max = 1./M_emax

    def _setup_grid(self, conf, emin, emax, Wmatrix, I):
        """
        This computes the parameter grid for the gridded gibbs approach
        """
        if conf.grid is None:
            conf.grid = .001
        if isinstance(conf.grid, int):
            conf.k = conf.grid
            conf.grid = np.linspace(emin, emax, num=conf.k)
            conf.grid_step = conf.grid[1] - conf.grid[0]
        elif isinstance(conf.grid, float):
            conf.grid_step = conf.grid
            conf.grid = np.arange(emin, emax, conf.grid_step)
            conf.k = len(conf.grid)
        elif isinstance(conf.grid, str):
            try:
                conf.grid = np.load(conf.grid)
                if conf.shape[1] == 2:
                    conf.grid = conf.grid.T
                conf.grid, conf.logdets = conf.grid
                return #break out early so we don't recompute the grid
            except Exception as e:
                Warn('Error in reading log determinant grid from file.'
                     ' Must be a (2,k) matrix, first row is parameter values,'
                     ' second row is log determinants.')
                raise e
        else:
            if conf.grid.shape[1] == 2:
                conf.grid = conf.grid.T
            conf.grid, conf.logdets = conf.grid
            return #break out early so we don't recompute the grid
        conf.grid = conf.grid[1:-1] #omit endpoints
        conf.logdets = np.asarray([splogdet(spar.csc_matrix(I - param * Wmatrix)) 
                                   for param in conf.grid])

    def _setup_data(self, **tuning):
        """
        This sets up the same example problem as in the Dong & Harris HSAR code. 
        """

        In = np.identity(self.state.N)
        Ij = np.identity(self.state.J)
        ##Prior specs
        M0 = tuning.pop('M0', np.zeros((self.state.p, 1)))
        T0 = tuning.pop('T0', np.identity(self.state.p) * 100)
        a0 = tuning.pop('a0', .01)
        b0 = tuning.pop('b0', .01)
        c0 = tuning.pop('c0', .01)
        d0 = tuning.pop('d0', .01)

        ##fixed matrix manipulations for MCMC loops
        XtX = spdot(self.state.X.T, self.state.X)
        XtXi = la.inv(XtX)
        T0inv = la.inv(T0)
        T0invM0 = spdot(T0inv, M0)
        DtD = spdot(self.state.Delta.T, self.state.Delta)

        ##unchanged posterior conditionals for sigma_e, sigma_u
        ce = self.state.N/2. + c0
        au = self.state.J/2. + a0

        rval = {k:v for k,v in dict(locals()).items() if k is not 'tuning'}
        return rval, tuning

    def _setup_initial_values(self, **configs):
        """
        function to set up initial values based on that stored in keyword
        dictionary passed to init
        """
        st = self.state
        if configs.pop('Betas', None) is None:
            st.Betas = np.zeros((st.p,1))
        if configs.pop('Thetas', None) is None:
            st.Thetas = np.zeros((st.J, 1))
        if configs.pop('Sigma2_u', None) is None:
            st.Sigma2_u = stats.invgamma.rvs(st.a0, scale=st.b0)
        if configs.pop('Sigma2_e', None) is None:
            st.Sigma2_e = stats.invgamma.rvs(st.c0, scale=st.d0)
        if configs.pop('Rho', None) is None:
            st.Rho = .5
        if configs.pop('Lambda', None) is None:
            st.Lambda = .5

    def sample(self, ndraws, pop=False):
        """
        Sample from the joint posterior distribution defined by all of the
        parameters in the gibbs sampler. 

        Parameters
        ----------
        ndraws      :   int
                        number of samples from the joint posterior density to
                        take
        pop         :   bool
                        whether to eject the trace from the sampler. If true,
                        this function will return a namespace containing the
                        results of the sampler during the run, and the sampler's
                        trace will be refreshed at the end. 

        Returns
        -------
        updates all values in place, may return trace of sampling run if pop is
        True
        """
        while ndraws > 0:
            self.draw()
            ndraws -= 1
        if pop:
            outdict = copy.deepcopy(self.trace.__dict__)
            outtrace = NS()
            outtrace.__dict__.update(outdict)
            del self.trace
            self.trace = NS()
            return outtrace

    def draw(self):
        """
        Take exactly one sample from the joint posterior distribution
        """
        sample(self)
        for param in self.traced_params:
            self.trace.__dict__[param].append(self.state.__dict__[param])

class HSAR(Base_HSAR):
    def __init__(self, y, X, W, M, 
                 Z=None, Delta=None, membership=None,
                 #data options:
                 sparse=True, transform='r', n_samples=1000, verbose=False,
                 **options):
        """
        The Dong-Harris multilevel HSAR model, which is a spatial autoregressive
        model with two levels. The first level has a simultaneous
        spatially-autoregressive effect. The second level has a
        spatially-correlated error term.

        Parameters
        ----------
        y               response
        X               lower covariates
        W               lower weights
        M               upper weights
        Z               optional upper covariates
        Delta           aggregation matrix classifing W into M
        membership      vector containing classification of W into M
        spars           bool, whether or not to keep weights in sparse forme
        transform       string, whether or not to row-standardize the weights
        n_samples       number of samples to draw
        verbose         bool, denoting whether to print tons of verbose messages
        configuration   parameters as defined in _setup_configs
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

        Wmat = W.sparse
        Mmat = M.sparse
        
        Delta, membership = verify.Delta_members(Delta, membership, N, J)
        
        # Data
        X = verify.covariates(X, W)
        
        self._verbose = verbose
        super(HSAR, self).__init__(y, X, Wmat, Mmat, Z, Delta, 
                                   n_samples=n_samples, **options)

def _setup():
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
    return data, y, X, W_low, W_up, membership, Z, test
