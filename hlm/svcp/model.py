from __future__ import division
import numpy as np
import copy
import scipy.linalg as scla
from scipy import stats 
from scipy.spatial import distance as d
from .utils import explode
from .sample import sample, H
from ..utils import Namespace as NS 

class SVCP(object):
    """
    A class to initialize a spatially-varying coefficient model

    Parameters to be estimated:
    Beta    : effects at each point, distributed around Mu _\beta
    Mus     : hierarchical mean of Beta effects
    T       : global covariance matrix for effects
    Phi     : spatial dependence parameter for effects
    Tau2    : prediction error for the model

    Hyperparameters as given:
    a0          : tau location hyperparameter
    b0          : tau scale hyperparameter
    v0          : T matrix degrees of freedom parameter
    Omega       : T matrix covariance prior
    mu0         : Mean beta hyperparameter
    sigma20     : Variance of beta hyperparameter
    phi_shape0  : Phi spatial dependence shape hyperparameter
    phi_rate0   : Phi spatial dependence rate hyperparameter
    phi_scale0  : Phi spatial dependence scale hyperparameter, overrides the
                  rate parameter. always 1/phi_rate0
    """
    def __init__(self,
                 #data parameters
                 y, X, coordinates, n_samples=1000, 
                 # prior configuration parameters
                 a0=2, b0=1, v0 = 3, Omega = None, mu0=0, sigma20 = None,
                 phi_shape0=.15, phi_rate0=.05, phi_scale0=None, 
                 #sampler starting values
                 Phi=None, T=None, Mus=None, Betas=None,
                 #Metropolis sampling parameters
                 tuning=0, initial_jump=5, adapt_step = 1.001, 
                 lower_ar=.4, upper_ar=.6, proposal=stats.norm,
                 #additional configuration parameters
                 extra_tracked_params = None, dmetric='euclidean'):
        n,p = X.shape
        Xs = X
        
        X = explode(X)


        self.state = NS() 
        self.trace = NS() 
        self.traced_params = ['Betas', 'Mus', 'T', 'Phi', 'Tau2'] 
        if extra_tracked_params is not None:
            self.traced_params.extend(extra_tracked_params)
        st = self.state
        
        for param in self.traced_params:
            self.trace.__dict__.update({param:[]})

        # the spatial param in wheeler & calder (2010) has a prior Ga(loc, rate)
        # but scipy is Ga(loc, scale). So, if user passes scale, convert to rate so
        # that we can keep the parameterization consistent with the article
        if phi_scale0 is not None and phi_rate0 == .05:
            phi_rate0 = 1 / phi_scale0

        st.y = y
        st.X = X
        st.Xs = Xs
        st.n = n
        st.p = p
       
        st.coordinates = coordinates
        st._dmetric = dmetric
        if isinstance(st._dmetric, str):
            st.pwds = d.squareform(d.pdist(st.coordinates, metric=st._dmetric))
        elif callable(st._dmetric):
            st.pwds = st._dmetric(st.coordinates)

        self._setup_priors(a0, b0, v0, Omega, mu0, sigma20, phi_shape0, phi_rate0)
        self._compute_invariants()

        self.configs.Phi = NS()
        self.configs.Phi.proposal = proposal
        self.configs.Phi.accepted = 0
        self.configs.Phi.rejected = 0
        self.configs.Phi.adapt_step = adapt_step
        self.configs.Phi.jump = initial_jump
        self.configs.Phi.ar_low = .4
        self.configs.Phi.ar_hi = .6
        if tuning > 0:
            self.configs.tuning = True
            self.configs.max_tuning = tuning
        else:
            self.configs.tuning = False
            self.configs.max_tuning = 0
        
        self.state._n_iterations = 0
        self.sample(n_samples)

    def _setup_priors(self, a0, b0, v0, Omega, mu0, sigma20, phi_shape0, phi_rate0):
        st = self.state
        st.a0 = a0
        st.b0 = b0
        st.v0 = v0
        if Omega is None:
            st.Ip = np.identity(st.p)
            st.Omega0 = .1 * st.Ip
        if type(mu0) in (float, int):
            st.mu0 = np.ones((st.p,1)) * mu0
        if sigma20 is None:
            st.sigma20 = 1000
        st.alpha0 = phi_shape0
        st.lambda0 = phi_rate0

    def _setup_initials(self, Phi, T, Mus, Betas):
        if T is None:
            means = np.zeros((self.state.p,))
            covm = np.identity((self.state.p)) * .00001
            self.state.T = np.random.multivariate_normal(means, covm)
        if Mus is None:
            self.state.Mus = np.zeros((1,self.state.p))
        if Betas is None:
            self.state.Betas = np.zeros((self.state.p * self.state.n, 1))
        if Phi is None:
            self.state.Phi = np.random.random()*np.max(self.state.pwds)

    def _compute_invariants(self):
        st = self.state
        st.In = np.identity(st.n)
        st.tau_dof = st.a0 + st.n/2
        st.T_dof = st.v0 + st.n
        st.mu0_cov = st.sigma20 * st.Ip
        st.mu0_cov_inv = scla.inv(st.mu0_cov)
        st.mu_kernel_prior = np.dot(st.mu0_cov_inv, st.mu0)
        st.XtX = np.dot(st.X.T, st.X)
        st.iota_n = np.ones((st.n,1))
        st.Xty = np.dot(st.X.T, st.y)

    def sample(self, ndraws, pop=False):
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
        sample(self)
        for param in self.traced_params:
            self.trace.__dict__[param].append(self.state.__dict__[param])
