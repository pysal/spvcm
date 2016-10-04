from __future__ import division
import numpy as np
import copy
import scipy.linalg as scla
import scipy.sparse as spar
from scipy import stats
from scipy.spatial import distance as d
from .utils import explode, nexp
from .sample import sample
from ...abstracts import Sampler_Mixin, Trace, Hashmap

class MSVCP(Sampler_Mixin):
    """
    A class to initialize a Multi-spatially-varying Coefficient model

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
                 Y, X=None, Z=None, coordinates, n_samples=1000,
                 # prior configuration parameters
                 tau2_a0=2, tau2_b0=1, mus_mean0=0, mus_cov0=None,
                 gammas_mean0=None, gammas_cov0=None,
                 phis_scale0_list=None, phis_rate0_list=None,
                 sigma2s_a0_list=None, sigma2s_b0_list=None,
                 #sampler starting values
                 phis_start=None, sigma2s_start=None,
                 mus_start=None, gammas_start=None,
                 zetas_start=None, tau2_start = None,
                 #Metropolis sampling parameters
                 tuning=0,
                 phi_jump=5, phi_adapt_step= 1.001,
                 phi_ar_low=.4, phi_ar_hi=.6,
                 phi_proposal=stats.norm,
                 #additional configuration parameters
                 extra_traced_params = None, dmetric='euclidean', verbose=False,
                 correlation_function=nexp,
                 **kwargs):
        n,p = X.shape

        self.state = Hashmap(**kwargs)
        self.traced_params = (['Gammas', 'Zetas', 'Mus',
                                'Sigma2s', 'Phis', 'Tau2'])
        if extra_traced_params is not None:
            self.traced_params.extend(extra_traced_params)
        self.trace = Trace(**{param:[] for param in self.traced_params})
        st = self.state
        self.state.correlation_function = correlation_function
        self.verbose = verbose
        

        st.Y = Y
        st.X = X
        st.Xc = explode_stack(X)
        st.n = n
        st.p = p
        if Z is not None:
            st.Z = Z
            st.pz = Z.shape[-1]
            self.state._has_Z = True
        else:
            del trace['Gammas']
            self.ZGammas = 0
            self.state._has_Z = False
        st.coordinates = coordinates
        st._dmetric = dmetric
        if isinstance(st._dmetric, str):
            st.pwds = d.squareform(d.pdist(st.coordinates, metric=st._dmetric))
        elif callable(st._dmetric):
            st.pwds = st._dmetric(st.coordinates)

        st.max_dist = st.pwds.max()
        st.pwds = st.pwds/st.max_dist

        self._setup_priors(tau2_a0, tau2_b0, mu0, mu_cov0,
                           phi_shape0_list, phi_rate0_list,
                           sigma2_a0_list, sigma2_b0_list,
                           gammas_mean0, gammas_cov0)
        self._setup_initials(phis_start, sigma2s_start,
                             mus_start, gammas_start,
                             zetas_start, tau2_start)
        self._compute_invariants()
        self._setup_configs(proposal=phi_proposal,
                            adapt_step = phi_adapt_step,
                            jump=phi_jump,
                            ar_low=phi_ar_low,
                            ar_hi = phi_ar_hi,
                            tuning=tuning
                            )
        self._verbose = verbose
        self.cycles = 0
        
        if n_samples > 0:
            try:
                self.sample(n_samples, n_jobs=n_jobs)
            except (np.linalg.LinAlgError, ValueError) as e:
                Warn('Encountered the following LinAlgError. '
                     'Model will return for debugging. \n {}'.format(e))

    def _setup_priors(self,tau2_a0, tau2_b0, mu_mean0, mu_cov0, gamma_mean0, gamma_cov0, phi_shape0_list, phi_rate0_list, sigma2_a0_list, sigma2_b0_list):
        st = self.state
        st.tau2_a0 = a0
        st.tau2_b0 = b0
        
        st.mu_cov0 = mu_cov0
        st.mu_mean0 = mu_mean0
        
        if st._has_Z:
            st.gamma_cov0 = gamma_cov0
            st.gamma_mean0 = gamma_mean0
            
        if phi_shape0_list is None:
            phi_shape0_list = [1]*st.p
        elif isinstance(phi_shape0_list, (float, int)):
            phi_shape0_list = [phi_shape0_list] * st.p
        st.phi_shape0_list = phi_shape0_list

        if phi_rate0_list is None:
           phi_rate0_list = ([phi_shape0 /
                                 ((-.5*st.pwds.max() / np.log(.05)))
                                 for phi_shape0 in phi_shape0_list])
        elif isinstance(phi_rate0_list, (float,int)):
            phi_rate0_list = [phi_rate0_list]*st.p
        st.phi_rate0_list = phi_rate0
        
        if sigma2_a0_list is None:
            sigma2_a0_list = [2] * st.p
        elif isinstance(sigma2_a0_list, (float,int)):
            sigma2_a0_list = [sigma2_a0_list]*st.p
        st.sigma2_a0_list = sigma2_a0_list
        
        if sigma2_b0_list is None:
            sigma2_b0_list = [2]*st.p
        elif isinstance(sigma2_b0_list, (float,int)):
            sigma2_b0_list = [sigma2_b0_list]*st.p
        st.sigma2_b0_list = sigma2_b0_list
            
    def _setup_configs(self, proposal, adapt_step, jump, ar_low, ar_hi, tuning, max_tuning):
        """
        Ensure that the sampling configuration is set accordingly.
        """
        self.configs = Hashmap(Phis=[Hashmap()]*self.state.p)
        
        for i,config in enumerate(self.configs.Phis):
            config.proposal = proposal[i] if isinstance(proposal, list) else proposal
            config.accepted = 0
            config.rejected = 0
            config.adapt_step = adapt_step[i] if isinstance(adapt_step, list) else adapt_step
            config.jump = jump[i] if isinstance(jump, list) else jump
            config.ar_low = ar_low if isinstance(ar_low, list) else ar_low
            config.ar_hi =  ar_hi if isinstance(ar_hi, list) else ar_hi
            if tuning > 0:
                self.configs.tuning = True
                config.max_tuning = tuning
            else:
                self.configs.tuning = False
                config.max_tuning = 0

    def _setup_initials(self, phis_start, sigma2s_start, mus_start, gammas_start, zetas_start, tau2_start):
        """
        set up initial values for the sampler
        """
        st = self.state
        
        if phis_start is None:
            phis_start = [3*phi_shape0_i / phi_rate0_i
                            for phi_shape0_i, phi_rate0_i
                            in zip(st.phi_shape0_list, st.phi_rate0_list)]
        self.state.Phi_list = phis_start
        if sigma2s_start is None:
            sigma2s_start = [1]*st.p
        self.state.Sigma2_list = sigma2s_start
        
        if mus_start is None:
            mus_start = np.zeros((1,st.p))
        self.state.Mus = mus_start

        if gammas_start is None:
            gammas_start = np.zeros((st.pz, 1))
        self.state.Gammas = gammas_start

        if zetas_start is None:
            zetas_start = np.zeros((st.n * st.p, 1))
        self.Zetas = zetas_start

        if tau2_start is None:
            tau2_start = 2
        self.state.Tau2 = tau2_start

    def _compute_invariants(self):
        st = self.state
        st.In = np.identity(st.n)
        st.XtX = st.X.T.dot(st.X)
        st.XctXc = st.Xc.T.dot(st.Xc)
        if st._has_Z:
            st.ZtZ = st.Z.T.dot(Z)

    def _finalize_invariants(self):
        st = self.state
        st.H_list = [st.correlation_function(phi, st.pwds)
                     for phi in st.Phi_list]
        st.Hi_list = [np.linalg.inv(Hj) for Hj in st.H_list]
        st.Zeta_list = np.array_split(st.Zetas, st.p).tolist()
        sigma2_an_list = [self.state.n/2.0 + a0
                          for a0 in st.sigma2_a0_list]
        tau2_an = self.state.n / 2.0 + st.tau2_a0
        st.mu_cov0i = np.linalg.inv(st.mu_cov0)
        if st.has_Z:
            st.gamma_cov0i = np.linalg.inv(st.gamma_cov0)
        st.XcZetas = st.Xc.dot(st.Zetas)
        st.H = scla.block_diag(*st.H_list)
        st.Hi = scla.block_diag(*st.Hi_list)

    def _iteration(self):
        mu_beta(self.state)
        tau(self.state)
        zeta(self.state)
        if self.state._has_Z:
            gamma(self.state)
        all_j(self.state)
