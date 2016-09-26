from __future__ import division

import numpy as np
import scipy.stats as stats
import copy

from numpy import linalg as la
from warnings import warn as Warn
from pysal.spreg.utils import sphstack, spdot
from .sample import sample
from ...abstracts import Sampler_Mixin, Hashmap, Trace
from ... import verify
from ...utils import speigen_range, splogdet, ind_covariance

SAMPLERS = ['Alphas', 'Betas', 'Sigma2', 'Tau2', 'Lambda', 'Rho']

class Base_Generic(Sampler_Mixin):
    """
    The class that actually ends up setting up the Generic model. Sets configs,
    data, truncation, and initial parameters, and then attempts to apply the
    sample function n_samples times to the state.
    """
    def __init__(self, Y, X, W, M, Delta, n_samples=1000, **_configs):
        
        skip_covariance = _configs.pop('skip_covariance', False)
        
        N, p = X.shape
        _N, J = Delta.shape
        n_jobs = _configs.pop('n_jobs', 1)
        self.state = Hashmap(**{'X':X, 'Y':Y, 'M':M, 'W':W, 'Delta':Delta,
                           'N':N, 'J':J, 'p':p })
        self.traced_params = copy.deepcopy(SAMPLERS)
        extras = _configs.pop('extra_traced_params', None)
        if extras is not None:
            self.traced_params.extend(extras)
        hashmaps = [{k:[] for k in self.traced_params}]*n_jobs
        self.trace = Trace(*hashmaps)
        leftovers = self._setup_data(**_configs)
        self._setup_configs(**leftovers)
        self._setup_truncation()
        self._setup_initial_values()

        if not skip_covariance:
            self.state.Psi_1 = ind_covariance
            self.state.Psi_2 = ind_covariance
            self._setup_covariance()

        
        self.cycles = 0
        
        if n_samples > 0:
            try:
                self.sample(n_samples, n_jobs=n_jobs)
            except (np.linalg.LinAlgError, ValueError) as e:
                Warn('Encountered the following LinAlgError. '
                     'Model will return for debugging. \n {}'.format(e))

    def _setup_data(self, **hypers):
        """
        This sets up the data and hyperparameters of the problem.
        If the hyperparameters are to be adjusted, pass them as keyword arguments.

        Arguments
        ----------
        None

        """
        In = np.identity(self.state.N)
        Ij = np.identity(self.state.J)
        ## Prior specifications
        Sigma2_a0 = hypers.pop('Sigma2_a0', .001)
        Sigma2_b0 = hypers.pop('Sigma2_b0', .001)
        Betas_cov0 = hypers.pop('Betas_cov0', np.eye(self.state.p) * 100)
        Betas_mean0 = hypers.pop('Betas_mean0', np.zeros((self.state.p, 1)))
        Tau2_a0 = hypers.pop('Tau2_a0', .001)
        Tau2_b0 = hypers.pop('Tau2_b0', .001)
        LogLambda0 = hypers.pop('LogLambda0', None)
        LogRho0 = hypers.pop('LogRho0', None)

        XtX = np.dot(self.state.X.T, self.state.X)
        DeltatDelta = np.dot(self.state.Delta.T, self.state.Delta)
        
        innovations = {k:v for k,v in dict(locals()).items() if k not in ['hypers', 'self']}
        self.state.update(innovations)
        return hypers
    
    def _finalize_invariants(self):
        """
        This computes derived properties of hyperparameters that do not change
        over iterations. This is called one time before sampling.
        """
        st = self.state
        st.Betas_cov0i = np.linalg.inv(st.Betas_cov0)
        st.Betas_covm = np.dot(st.Betas_cov0, st.Betas_mean0)
        st.Sigma2_an = self.state.N / 2 + st.Sigma2_a0
        st.Tau2_an = self.state.J / 2 + st.Tau2_a0
        if st.LogLambda0 is None:
            eigenrange = st.Lambda_max - st.Lambda_min
            Lambda_logprior = np.log(1/eigenrange)
            def LogLambda0(value):
                return Lambda_logprior
            st.LogLambda0 = LogLambda0
        if st.LogRho0 is None:
            eigenrange = st.Rho_max - st.Rho_min
            Rho_logprior = np.log(1/eigenrange)
            def LogRho0(value):
                return Rho_logprior
            st.LogRho0 = LogRho0

    def _setup_configs(self, #would like to make these keyword only using *
                 #multi-parameter options
                 tuning=0,
                 #spatial parameter grid sample configurations:
                 rho_jump=.5, rho_ar_low=.4, rho_ar_hi=.6,
                 rho_proposal=stats.norm, rho_adapt_step=1.01,
                 #spatial parameter grid sample configurations:
                 lambda_jump=.5, lambda_ar_low=.4, lambda_ar_hi=.6,
                 lambda_proposal=stats.norm, lambda_adapt_step=1.01):
        """
        Omnibus function to assign configuration parameters to the correct
        configuration namespace
        """
        self.configs = Hashmap()
        self.configs.Rho = Hashmap(jump = rho_jump, ar_low = rho_ar_low,
                                   ar_hi = rho_ar_hi,
                                   proposal = rho_proposal,
                                   adapt_step = rho_adapt_step,
                                   accepted = 0, rejected = 0,
                                   max_adapt = tuning,
                                   adapt = tuning > 0)
        self.configs.Lambda = Hashmap(jump = lambda_jump,
                                      ar_low = lambda_ar_low,
                                      ar_hi = lambda_ar_hi,
                                      proposal = lambda_proposal,
                                      adapt_step = lambda_adapt_step,
                                      accepted = 0, rejected = 0,
                                      max_adapt = tuning,
                                      adapt = tuning > 0)

    def _setup_truncation(self):
        """
        This computes truncations for the spatial parameters.
    
        If configs.truncate is set to 'eigs', computes the eigenrange of the two
        spatial weights matrices using speigen_range

        If configs.truncate is set to 'stable', sets the truncation to -1,1
        
        If a tuple is passed to truncate, then this will truncate the
        distribution according to this tuple
        """
        st = self.state
        W_emin, W_emax = speigen_range(st.W)
        st.Rho_min = 1./W_emin
        st.Rho_max = 1./W_emax
        M_emin, M_emax = speigen_range(st.M)
        st.Lambda_min = 1./M_emin
        st.Lambda_max = 1./M_emax

    def _setup_initial_values(self):
        """
        Set abrbitrary starting values for the Metropolis sampler
        """
        Betas = np.zeros((self.state.p ,1))
        Alphas = np.zeros((self.state.J, 1))
        Sigma2 = 4
        Tau2 = 4
        Rho = -1.0 / (self.state.N - 1)
        Lambda = -1.0 / (self.state.J - 1)
        DeltaAlphas = np.dot(self.state.Delta, Alphas)
        XBetas = np.dot(self.state.X, Betas)
        
        innovations = {k:v for k,v in dict(locals()).items() if k not in ['self']}
        self.state.update(innovations)

    def _setup_covariance(self):
        """
        Set up covariance, depending on model form. If this is SMA, the
        utils.sma_covariance function will be used. If this is SAR-error, the
        utils.ser_covariance function will be used.
        """
        st = self.state
        st.PsiRho = st.Psi_1(st.Rho, st.W)
        st.PsiLambda = st.Psi_2(st.Lambda, st.M)

        st.PsiSigma2 = st.PsiRho.dot(st.In * st.Sigma2)
        st.PsiSigma2i = la.inv(st.PsiSigma2)
        st.PsiTau2 = st.PsiLambda.dot(st.Ij * st.Tau2)
        st.PsiTau2i = la.inv(st.PsiTau2)
        
        st.PsiRhoi = la.inv(st.PsiRho)
        st.PsiLambdai = la.inv(st.PsiLambda)
    
    _sample = sample

class Generic(Base_Generic):
    """
    The class that intercepts & validates input
    """
    def __init__(self, Y, X, W, M, Z=None, Delta=None, membership=None,
                 #data options
                 transform ='r', n_samples=1000, verbose=False,
                 **options):
        W,M = verify.weights(W,M, transform=transform)
        self.M = M

        N,_ = X.shape
        J = M.n
        Mmat = M.sparse
        Wmat = W.sparse

        Delta, membership = verify.Delta_members(Delta, membership, N, J)

        X = verify.covariates(X)

        self._verbose = verbose
        if Z is not None:
            Z = Delta.dot(Z)
            X = np.hstack((X,Z))
        super(Generic, self).__init__(Y, X, Wmat, Mmat, Delta, n_samples,
                **options)
