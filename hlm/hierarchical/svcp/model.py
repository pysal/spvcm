from __future__ import division
import numpy as np
import copy
import scipy.linalg as scla
import scipy.sparse as spar
from scipy import stats 
from scipy.spatial import distance as d
from .utils import explode, nexp
from .sample import sample_phi
from ...abstracts import Sampler_Mixin, Trace, Hashmap
from ...utils import chol_mvn

class SVCP(Sampler_Mixin):
    """
    A class to initialize a spatially-varying coefficient model

    Parameters to be estimated:
    Beta    : effects at each point, distributed around Mu _\beta
    Mus     : hierarchical mean of Beta effects
    T       : global covariance matrix for effects
    Phi     : spatial dependence parameter for effects
    Tau2    : prediction error for the model

    The distance matrix will be divided by the maximum distance,
    so that it will vary between 0 and 1. 

    Hyperparameters as given:
    a0          : tau location hyperparameter
    b0          : tau scale hyperparameter
    v0          : T matrix degrees of freedom parameter
    Omega       : T matrix covariance prior
    mu0         : Mean beta hyperparameter
    mu_cov0      : Beta hyperparameter for Covariance
    phi_shape0  : Phi spatial dependence shape hyperparameter
    phi_rate0   : Phi spatial dependence rate hyperparameter
    phi_scale0  : Phi spatial dependence scale hyperparameter, overrides the
                  rate parameter. always 1/phi_rate0
    """
    def __init__(self,
                 #data parameters
                 Y, X, coordinates, n_samples=1000, 
                 # prior configuration parameters
                 a0=2, b0=1, v0 = 3, Omega = None, mu0=0, mu_cov0=None, 
                 phi_shape0=None, phi_rate0=None, phi_scale0=None, 
                 #sampler starting values
                 Phi=None, T=None, Mus=None, Betas=None,
                 #Metropolis sampling parameters
                 tuning=0, phi_jump=5, phi_adapt_step = 1.001, 
                 phi_ar_low=.4, phi_ar_hi=.6, phi_proposal=stats.norm,
                 #additional configuration parameters
                 extra_traced_params = None, dmetric='euclidean', verbose=False, 
                 correlation_function=nexp,
                 **kwargs):
        n,p = X.shape
        Xs = X
        
        X = explode(X)


        self.state = Hashmap(**kwargs) 
        self.traced_params = ['Betas', 'Mus', 'T', 'Phi', 'Tau2'] 
        if extra_traced_params is not None:
            self.traced_params.extend(extra_traced_params)
        self.trace = Trace(**{param:[] for param in self.traced_params})
        st = self.state
        self.state.correlation_function = correlation_function
        self.verbose = verbose
        
        # the spatial param in wheeler & calder (2010) has a prior Ga(loc, rate)
        # but scipy is Ga(loc, scale). So, if user passes scale, convert to rate so
        # that we can keep the parameterization consistent with the article
        if phi_scale0 is not None and phi_rate0 is None:
            phi_rate0 = 1 / phi_scale0

        st.Y = Y
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

        st.max_dist = st.pwds.max()
        st.pwds = st.pwds/st.max_dist


        self._setup_priors(a0, b0, v0, Omega, mu0, mu_cov0, phi_shape0, phi_rate0)
        self._setup_initials(Phi, T, Mus, Betas)
        self._compute_invariants()
        self.configs = Hashmap() 
        self.configs.Phi = Hashmap(proposal=phi_proposal, accepted=0, rejected=0,
                                   adapt_step = phi_adapt_step, jump=phi_jump,
                                   ar_low = phi_ar_low, ar_hi = phi_ar_hi,
                                   max_tuning = tuning)
        self.configs.tuning = tuning > 0
        
        self._verbose = verbose
        self.cycles = 0
        
        if n_samples > 0:
            try:
                self.sample(n_samples, n_jobs=n_jobs)
            except (np.linalg.LinAlgError, ValueError) as e:
                Warn('Encountered the following LinAlgError. '
                     'Model will return for debugging. \n {}'.format(e))

    def _setup_priors(self, a0, b0, v0, Omega, mu0, mu_cov0, phi_shape0, phi_rate0):
        st = self.state
        st.a0 = a0
        st.b0 = b0
        st.v0 = v0
        if Omega is None:
            st.Ip = np.identity(st.p)
            st.Omega0 = .1 * st.Ip
        if type(mu0) in (float, int):
            st.mu0 = np.ones((st.p,1)) * mu0
        if mu_cov0 is None:
            st.mu_cov0 = 1000*st.Ip
        if phi_shape0 is None:
           st.phi_shape0 = 1
        else:
            st.phi_shape0 = phi_shape0
        if phi_rate0 is None:
           st.phi_rate0 = st.phi_shape0 / ((-.5*st.pwds.max() / np.log(.05))) 
        else:
            st.phi_rate0 = phi_rate0

    def _setup_initials(self, Phi, T, Mus, Betas):
        if T is None:
            means = np.zeros((self.state.p,))
            covm = np.identity((self.state.p)) * .00001
            T = np.random.multivariate_normal(means, covm)
        self.state.T = T
        if Mus is None:
            Mus = np.zeros((1,self.state.p))
        self.state.Mus = Mus
        if Betas is None:
            Betas = np.zeros((self.state.p * self.state.n, 1))
        self.state.Betas = Betas
        if Phi is None:
            Phi = 3*self.state.phi_shape0 / self.state.phi_rate0
        self.state.Phi = Phi

    def _compute_invariants(self):
        st = self.state
        st.In = np.identity(st.n)
        st.Tau_dof = st.a0 + st.n/2
        st.T_dof = st.v0 + st.n
        st.XtX = st.X.T.dot(st.X)
        st.iota_n = np.ones((st.n,1))
        st.Xty = st.X.T.dot(st.Y)
        st.np2n = np.zeros((st.n * st.p, st.n))
        for i in range(st.n):
            st.np2n[i*st.p:(i+1)*st.p, i] = 1
        st.np2p = np.vstack([np.eye(st.p) for _ in range(st.n)])

    def _finalize_invariants(self):
        st = self.state
        st.mu_cov0_inv = scla.inv(st.mu_cov0)
        st.mu_kernel_prior = np.dot(st.mu_cov0_inv, st.mu0)
    

    def _iteration(self):
        """
        Conduct one iteration of a Gibbs sampler for the self using the state
        provided. 
        """
        st = self.state
        
        ## Tau, EQ 3 in appendix of Wheeler & Calder
        ## Inverse Gamma w/ update to scale, no change to dof
        y_Xbeta = st.Y - st.X.dot(st.Betas)
        scale = st.b0 + .5 * y_Xbeta.T.dot(y_Xbeta)
        st.Tau2 = stats.invgamma.rvs(st.Tau_dof, scale=scale)

        ##covariance: T, EQ 4 in appendix of Wheeler & Calder
        ## inverse wishart w/ update to covariance matrix, no change to dof
        st.H = st.correlation_function(st.Phi, st.pwds)
        st.Hinv = scla.inv(st.H)
        st.tiled_Hinv = np.linalg.multi_dot([st.np2n, st.Hinv, st.np2n.T])
        st.tiled_Mus = np.kron(st.iota_n, st.Mus.reshape(-1,1))
        st.info = (st.Betas - st.tiled_Mus).dot((st.Betas - st.tiled_Mus).T)
        st.kernel = np.multiply(st.tiled_Hinv, st.info) 
        st.covm_update = np.linalg.multi_dot([st.np2p.T, st.kernel, st.np2p])
        st.T = stats.invwishart.rvs(df=st.T_dof, scale=(st.covm_update + st.Omega0))

        ##mean hierarchical effects: mu_\beta, in EQ 5 of Wheeler & Calder
        ##normal with both a scale and a location update, priors don't change
        #compute scale of mu_\betas
        st.Sigma_beta = np.kron(st.H, st.T)
        st.Psi = np.linalg.multi_dot((st.X, st.Sigma_beta, st.X.T)) + st.Tau2 * st.In
        Psi_inv = scla.inv(st.Psi)
        S_notinv_update = np.linalg.multi_dot((st.Xs.T, Psi_inv, st.Xs))
        S = scla.inv(st.mu_cov0_inv + S_notinv_update)
        
        #compute location of mu_\betas
        mkernel_update = np.linalg.multi_dot((st.Xs.T, Psi_inv, st.Y))  
        st.Mu_means = np.dot(S, (mkernel_update + st.mu_kernel_prior))
        
        #draw them using cholesky decomposition: N(m, Sigma) = m + chol(Sigma).N(0,1)
        st.Mus = chol_mvn(st.Mu_means, S) 
        st.tiled_Mus = np.kron(st.iota_n, st.Mus)

        ##effects \beta, in equation 6 of Wheeler & Calder
        ##Normal with an update to both scale and location, priors don't change
        
        #compute scale of betas
        st.Tinv = scla.inv(st.T)
        st.kronHiTi = np.kron(st.Hinv, st.Tinv)
        Ai = st.XtX / st.Tau2 + st.kronHiTi
        A = scla.inv(Ai)
        
        #compute means of betas
        C = st.Xty / st.Tau2 + np.dot(st.kronHiTi, st.tiled_Mus)
        st.Beta_means = np.dot(A, C)
        st.Beta_cov = A
        
        #draw them using cholesky decomposition
        st.Betas = chol_mvn(st.Beta_means, st.Beta_cov)

        # local nonstationarity parameter Phi, in equation 7 in Wheeler & Calder
        # sample using metropolis
        st.Phi = sample_phi(self)
