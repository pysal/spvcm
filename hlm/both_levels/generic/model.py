from __future__ import division

import numpy as np
import scipy.stats as stats
import copy

from numpy import linalg as la
from warnings import warn as Warn
from pysal.spreg.utils import sphstack, spdot
from .sample import sample_spatial, logp_rho, logp_lambda
from ...abstracts import Sampler_Mixin, Hashmap, Trace
from ... import verify
from ... import priors
from ...utils import speigen_range, splogdet, ind_covariance, chol_mvn

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
        Log_Lambda0 = hypers.pop('Log_Lambda0', None)
        Log_Rho0 = hypers.pop('Log_Rho0', None)

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
        if st.Log_Lambda0 is None:
            st.Log_Lambda0 = priors.constant
        if st.Log_Rho0 is None:
            st.Log_Rho0 = priors.constant

    def _setup_configs(self, #would like to make these keyword only using *
                 #multi-parameter options
                 tuning=0,
                 #spatial parameter grid sample configurations:
                 Rho_jump=.5, Rho_ar_low=.4, Rho_ar_hi=.6,
                 Rho_proposal=stats.norm, Rho_adapt_step=1.01,
                 #spatial parameter grid sample configurations:
                 Lambda_jump=.5, Lambda_ar_low=.4, Lambda_ar_hi=.6,
                 Lambda_proposal=stats.norm, Lambda_adapt_step=1.01):
        """
        Omnibus function to assign configuration parameters to the correct
        configuration namespace
        """
        self.configs = Hashmap()
        self.configs.Rho = Hashmap(jump = Rho_jump, ar_low = Rho_ar_low,
                                   ar_hi = Rho_ar_hi,
                                   proposal = Rho_proposal,
                                   adapt_step = Rho_adapt_step,
                                   accepted = 0, rejected = 0,
                                   max_adapt = tuning,
                                   adapt = tuning > 0)
        self.configs.Lambda = Hashmap(jump = Lambda_jump,
                                      ar_low = Lambda_ar_low,
                                      ar_hi = Lambda_ar_hi,
                                      proposal = Lambda_proposal,
                                      adapt_step = Lambda_adapt_step,
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
        Set arbitrary starting values for the Metropolis sampler
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
    

    def _iteration(self):
        st = self.state
    
        ### Sample the Beta conditional posterior
        ### P(beta | . ) \propto L(Y|.) \dot P(\beta)
        ### is
        ### N(Sb, S) where
        ### S = (X' Sigma^{-1}_Y X + S_0^{-1})^{-1}
        ### b = X' Sigma^{-1}_Y (Y - Delta Alphas) + S^{-1}\mu_0
        covm_update = st.X.T.dot(st.PsiRhoi).dot(st.X) / st.Sigma2
        covm_update += st.Betas_cov0i
        covm_update = la.inv(covm_update)
    
        resids = st.Y - st.Delta.dot(st.Alphas)
        XtSresids = st.X.T.dot(st.PsiRhoi).dot(resids) / st.Sigma2
        mean_update = XtSresids + st.Betas_cov0i.dot(st.Betas_mean0)
        mean_update = np.dot(covm_update, mean_update)
        st.Betas = chol_mvn(mean_update, covm_update)
        st.XBetas = np.dot(st.X, st.Betas)
    
        ### Sample the Random Effect conditional posterior
        ### P( Alpha | . ) \propto L(Y|.) \dot P(Alpha | \lambda, Tau2)
        ###                               \dot P(Tau2) \dot P(\lambda)
        ### is
        ### N(Sb, S)
        ### Where
        ### S = (Delta'Sigma_Y^{-1}Delta + Sigma_Alpha^{-1})^{-1}
        ### b = (Delta'Sigma_Y^{-1}(Y - X\beta) + 0)
        covm_update = st.Delta.T.dot(st.PsiRhoi).dot(st.Delta) / st.Sigma2
        covm_update += st.PsiLambdai / st.Tau2
        covm_update = la.inv(covm_update)
    
        resids = st.Y - st.XBetas
        mean_update = st.Delta.T.dot(st.PsiRhoi).dot(resids) / st.Sigma2
        mean_update = np.dot(covm_update, mean_update)
        st.Alphas = chol_mvn(mean_update, covm_update)
        st.DeltaAlphas = np.dot(st.Delta, st.Alphas)
    
        ### Sample the Random Effect aspatial variance parameter
        ### P(Tau2 | .) \propto L(Y|.) \dot P(\Alpha | \lambda, Tau2)
        ###                            \dot P(Tau2) \dot P(\lambda)
        ### is
        ### IG(J/2 + a0, u'(\Psi(\lambda))^{-1}u * .5 + b0)
        bn = st.Alphas.T.dot(st.PsiLambdai).dot(st.Alphas) * .5 + st.Tau2_b0
        st.Tau2 = stats.invgamma.rvs(st.Tau2_an, scale=bn)
        
        ### Sample the response aspatial variance parameter
        ### P(Sigma2 | . ) \propto L(Y | .) \dot P(Sigma2)
        ### is
        ### IG(N/2 + a0, eta'Psi(\rho)^{-1}eta * .5 + b0)
        ### Where eta is the linear predictor, Y - X\beta + \DeltaAlphas
        eta = st.Y - st.XBetas - st.DeltaAlphas
        bn = eta.T.dot(st.PsiRhoi).dot(eta) * .5 + st.Sigma2_b0
        st.Sigma2 = stats.invgamma.rvs(st.Sigma2_an, scale=bn)
    
        ### Sample the spatial components using metropolis-hastings
        ### P(Psi(\lambda) | .) \propto L(Y | .) \dot P(\lambda)
        ### is
        ### |Psi(lambda)|^{-1/2} exp(1/2(Alphas'Psi(lambda)^{-1}Alphas * Tau2^{-1}))
        ###  * 1/(emax-emin)
        st.Rho = sample_spatial(self.configs.Rho, st.Rho, st,
                                logp=logp_rho)
        
        st.PsiRho = st.Psi_1(st.Rho, st.W)
        st.PsiSigma2 = st.PsiRho.dot(st.In*st.Sigma2)
        st.PsiSigma2i = la.inv(st.PsiSigma2)
        st.PsiRhoi = la.inv(st.PsiRho)
            
        ### P(Psi(\rho) | . ) \propto L(Y | .) \dot P(\rho)
        ### is
        ### |Psi(rho)|^{-1/2} exp(1/2(eta'Psi(rho)^{-1}eta * Sigma2^{-1})) * 1/(emax-emin)
        st.Lambda = sample_spatial(self.configs.Lambda, st.Lambda, st,
                                   logp=logp_lambda)
        st.PsiLambda = st.Psi_2(st.Lambda, st.M)
        st.PsiTau2 = st.PsiLambda.dot(st.Ij * st.Tau2)
        st.PsiTau2i = la.inv(st.PsiTau2)
        st.PsiLambdai = la.inv(st.PsiLambda)
        
        

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
