from __future__ import division

import numpy as np
import scipy.stats as stats
import copy

from numpy import linalg as la
from warnings import warn as Warn
from .sample import logp_rho_cov, logp_lambda_cov
from ...abstracts import Sampler_Mixin, Hashmap, Trace
from ... import verify
from ... steps import Metropolis, Slice
from ... import priors
from ...utils import speigen_range, ind_covariance, chol_mvn
from pysal.spreg.utils import spdot

SAMPLERS = ['Alphas', 'Betas', 'Sigma2', 'Tau2', 'Lambda', 'Rho']

class Base_Generic(Sampler_Mixin):
    """
    The class that actually ends up setting up the Generic model. Sets configs,
    data, truncation, and initial parameters, and then attempts to apply the
    sample function n_samples times to the state.
    """
    def __init__(self, Y, X, W, M, Delta,
                 n_samples=1000, n_jobs=1,
                 extra_traced_params = None,
                 priors=None,
                 starting_values=None,
                 configs=None,
                 truncation=None):
        super(Base_Generic, self).__init__()
        
        N, p = X.shape
        _N, J = Delta.shape
        self.state = Hashmap(**{'X':X, 'Y':Y, 'M':M, 'W':W, 'Delta':Delta,
                           'N':N, 'J':J, 'p':p })
        self.traced_params = copy.deepcopy(SAMPLERS)
        if extra_traced_params is not None:
            self.traced_params.extend(extra_traced_params)
        hashmaps = [{k:[] for k in self.traced_params}]*n_jobs
        self.trace = Trace(*hashmaps)
        
        if priors is None:
            priors = dict()
        if starting_values is None:
            starting_values = dict()
        if configs is None:
            configs = dict()
        if truncation is None:
            truncation = dict()
        
        self._setup_priors(**priors)
        self._setup_configs(**configs)
        self._setup_truncation(**truncation)
        self._setup_starting_values(**starting_values)
        
        ## Covariance, computing the starting values
        self.state.Psi_1 = ind_covariance
        self.state.Psi_1i = ind_covariance
        self.state.Psi_2 = ind_covariance
        self.state.Psi_2i = ind_covariance
        
        self.cycles = 0
        
        if n_samples > 0:
            try:
                self.sample(n_samples, n_jobs=n_jobs)
            except (np.linalg.LinAlgError, ValueError) as e:
                Warn('Encountered the following LinAlgError. '
                     'Model will return for debugging. \n {}'.format(e))

    def _setup_priors(self, Betas_cov0 = None, Betas_mean0=None,
                      Sigma2_a0 = .001, Sigma2_b0 = .001,
                      Tau2_a0 = .001, Tau2_b0 = .001,
                      Log_Lambda0 = priors.constant,
                      Log_Rho0 = priors.constant):
        """
        This sets up the data and hyperparameters of the problem.
        If the hyperparameters are to be adjusted, pass them as keyword arguments.

        Arguments
        ----------
        None

        """
        st = self.state
        st.Sigma2_a0 = Sigma2_a0
        st.Sigma2_b0 = Sigma2_b0
        if Betas_cov0 is None:
            Betas_cov0 = np.eye(self.state.p) * 100
        if Betas_mean0 is None:
            Betas_mean0 = np.zeros((self.state.p, 1))
        st.Betas_cov0 = Betas_cov0
        st.Betas_mean0 = Betas_mean0
        st.Tau2_a0 = .001
        st.Tau2_b0 = .001
        st.Log_Lambda0 = Log_Lambda0
        st.Log_Rho0 = Log_Rho0

    def _finalize(self):
        """
        This computes derived properties of hyperparameters that do not change
        over iterations. This is called one time before sampling.
        """
        st = self.state
        
        st.In = np.eye(st.N)
        st.Ij = np.eye(st.J)
        
        ## Derived factors from the prior
        st.Betas_cov0i = np.linalg.inv(st.Betas_cov0)
        st.Betas_covm = np.dot(st.Betas_cov0, st.Betas_mean0)
        st.Sigma2_an = self.state.N / 2 + st.Sigma2_a0
        st.Tau2_an = self.state.J / 2 + st.Tau2_a0
        
        st.PsiRhoi = st.Psi_1i(st.Rho, st.W)
        st.PsiLambdai = st.Psi_2i(st.Lambda, st.M)
        
        ## Data invariants
        st.XtX = np.dot(self.state.X.T, self.state.X)
        st.DeltatDelta = np.dot(self.state.Delta.T, self.state.Delta)

        st.DeltaAlphas = np.dot(st.Delta, st.Alphas)
        st.XBetas = np.dot(st.X, st.Betas)
            
        st.initial_values = Hashmap(Betas = st.Betas, Alphas=st.Alphas, Rho=st.Rho, Lambda=st.Lambda,
                                    Sigma2=st.Sigma2, Tau2=st.Tau2)


    def _setup_configs(self, Lambda_method = 'met', Lambda_configs = None,
                             Rho_method = 'met', Rho_configs = None,
                             **uncaught):
        """
        Omnibus function to assign configuration parameters to the correct
        configuration namespace
        """
        if Lambda_configs is None:
            Lambda_configs = dict()
        if Rho_configs is None:
            Rho_configs = dict()
            
        if uncaught != dict():
            if 'method' in uncaught:
                Lambda_method = Rho_method = method
            Lambda_configs = copy.deepcopy(uncaught)
            Rho_configs = copy.deepcopy(uncaught)

        if Lambda_method.lower().startswith('met'):
            Lambda_method = Metropolis
            Lambda_configs['jump'] = Lambda_configs.pop('jump', .5)
            Lambda_configs['max_tuning'] = Lambda_configs.pop('tuning',0)
        elif Lambda_method.lower().startswith('slice'):
            Lambda_method = Slice
            Lambda_configs['width'] = Lambda_configs.pop('width', .5)
        else:
            raise Exception('Sample method for Lambda not understood:\n{}'
                            .format(Lambda_method))
        if Rho_method.lower().startswith('met'):
            Rho_method = Metropolis
            Rho_configs['jump'] = Rho_configs.pop('jump', .5)
            Rho_configs['max_tuning'] = Rho_configs.pop('tuning',0)

        elif Rho_method.lower().startswith('slice'):
            Rho_method = Slice
            Rho_configs['width'] = Rho_configs.pop('width', .5)
        else:
            raise Exception('Sample method for Rho not understood:\n{}'
                            .format(Rho_method))
        print('Rho method, configs: {}'.format(Rho_method, Rho_configs))
        print('Lambda method, configs: {}'.format(Lambda_method, Lambda_configs))

        self.configs = Hashmap()
        self.configs.Rho = Rho_method('Rho', logp_rho_cov, **Rho_configs)
        self.configs.Lambda = Lambda_method('Lambda', logp_lambda_cov, **Lambda_configs)

    def _setup_truncation(self, Rho_min=None, Rho_max = None,
                          Lambda_min = None, Lambda_max = None):
        """
        This computes truncations for the spatial parameters.
    
        If configs.truncate is set to 'eigs', computes the eigenrange of the two
        spatial weights matrices using speigen_range

        If configs.truncate is set to 'stable', sets the truncation to -1,1
        
        If a tuple is passed to truncate, then this will truncate the
        distribution according to this tuple
        """
        st = self.state
        if hasattr(st, 'W'):
            if (Rho_min is None) or (Rho_max is None):
                W_emin, W_emax = speigen_range(st.W)
            if (Rho_min is None):
                Rho_min = 1./W_emin
            if (Rho_max is None):
                Rho_max = 1./W_emax
            st.Rho_min = Rho_min
            st.Rho_max = Rho_max
        if hasattr(st, 'M'):
            if (Lambda_min is None) or (Lambda_max is None):
                M_emin, M_emax = speigen_range(st.M)
            if (Lambda_min is None):
                Lambda_min = 1./M_emin
            if (Lambda_max is None):
                Lambda_max = 1./M_emax
            st.Lambda_min = Lambda_min
            st.Lambda_max = Lambda_max

    def _setup_starting_values(self, Betas = None, Alphas = None,
                               Sigma2 = 4, Tau2 = 4,
                               Rho = None, Lambda = None):
        """
        Set arbitrary starting values for the Metropolis sampler
        """
        st = self.state
        if Betas is None:
            Betas = np.zeros((self.state.p, 1))
        if Alphas is None:
            Alphas = np.zeros((self.state.J, 1))
        st.Betas = Betas
        st.Alphas = Alphas
        st.Sigma2 = Tau2
        st.Tau2 = Sigma2
        if Rho is None:
            Rho = -1.0 / (self.state.N - 1)
        if Lambda is None:
            Lambda = -1.0 / (self.state.J - 1)
        st.Rho = Rho
        st.Lambda = Lambda
        
    def _iteration(self):
        """
        Compute a single iteration of the sampler.
        This steps through all parameter updates exactly once.
        """
        st = self.state
    
        ### Sample the Beta conditional posterior
        ### P(beta | . ) \propto L(Y|.) \dot P(\beta)
        ### is
        ### N(Sb, S) where
        ### S = (X' Sigma^{-1}_Y X + S_0^{-1})^{-1}
        ### b = X' Sigma^{-1}_Y (Y - Delta Alphas) + S^{-1}\mu_0
        covm_update = spdot(st.X.T, spdot(st.PsiRhoi, st.X)) / st.Sigma2
        covm_update += st.Betas_cov0i
        covm_update = np.asarray(covm_update)
        covm_update = la.inv(covm_update)
    
        resids = st.Y - st.Delta.dot(st.Alphas)
        XtSresids = spdot(st.X.T, spdot(st.PsiRhoi, resids)) / st.Sigma2
        mean_update = XtSresids + st.Betas_cov0i.dot(st.Betas_mean0)
        mean_update = np.asarray(mean_update)
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
        covm_update = spdot(st.Delta.T, spdot(st.PsiRhoi, st.Delta)) / st.Sigma2
        covm_update += st.PsiLambdai / st.Tau2
        covm_update = np.asarray(covm_update)
        covm_update = la.inv(covm_update)
    
        resids = st.Y - st.XBetas
        mean_update = spdot(st.Delta.T, spdot(st.PsiRhoi, resids)) / st.Sigma2
        mean_update = np.asarray(mean_update)
        mean_update = np.dot(covm_update, mean_update)
        st.Alphas = chol_mvn(mean_update, covm_update)
        st.DeltaAlphas = np.dot(st.Delta, st.Alphas)
    
        ### Sample the Random Effect aspatial variance parameter
        ### P(Tau2 | .) \propto L(Y|.) \dot P(\Alpha | \lambda, Tau2)
        ###                            \dot P(Tau2) \dot P(\lambda)
        ### is
        ### IG(J/2 + a0, u'(\Psi(\lambda))^{-1}u * .5 + b0)
        bn = spdot(st.Alphas.T, spdot(st.PsiLambdai, st.Alphas)) * .5 + st.Tau2_b0
        st.Tau2 = stats.invgamma.rvs(st.Tau2_an, scale=bn)
        
        ### Sample the response aspatial variance parameter
        ### P(Sigma2 | . ) \propto L(Y | .) \dot P(Sigma2)
        ### is
        ### IG(N/2 + a0, eta'Psi(\rho)^{-1}eta * .5 + b0)
        ### Where eta is the linear predictor, Y - X\beta + \DeltaAlphas
        eta = st.Y - st.XBetas - st.DeltaAlphas
        bn = spdot(eta.T, spdot(st.PsiRhoi, eta)) * .5 + st.Sigma2_b0
        st.Sigma2 = stats.invgamma.rvs(st.Sigma2_an, scale=bn)
    
        ### Sample the spatial components using metropolis-hastings
        ### P(Psi(\lambda) | .) \propto L(Y | .) \dot P(\lambda)
        ### is
        ### |Psi(lambda)|^{-1/2} exp(1/2(Alphas'Psi(lambda)^{-1}Alphas * Tau2^{-1}))
        ###  * 1/(emax-emin)
        st.Rho = self.configs.Rho(st)
        
        st.PsiRhoi = st.Psi_1i(st.Rho, st.W) #BtB or CiCti
            
        ### P(Psi(\rho) | . ) \propto L(Y | .) \dot P(\rho)
        ### is
        ### |Psi(rho)|^{-1/2} exp(1/2(eta'Psi(rho)^{-1}eta * Sigma2^{-1})) * 1/(emax-emin)
        st.Lambda = self.configs.Lambda(st)

        st.PsiLambdai = st.Psi_2i(st.Lambda, st.M)

class Generic(Base_Generic):
    """
    The class that intercepts & validates input
    """
    def __init__(self, Y, X, W, M, Z=None, Delta=None, membership=None,
                 #data options
                 transform ='r', verbose=False,
                 n_samples=1000, n_jobs=1,
                 extra_traced_params = None,
                 priors=None,
                 starting_values=None,
                 configs=None,
                 truncation=None,
                 center=True,
                 scale=False):
        W,M = verify.weights(W,M, transform=transform)
        self.M = M
        
        
        N,_ = X.shape
        J = M.n
        Mmat = M.sparse
        Wmat = W.sparse
        Delta, membership = verify.Delta_members(Delta, membership, N, J)

        
        if Z is not None:
            Z = Delta.dot(Z)
            X = np.hstack((X,Z))
        if center:
            X = verify.center(X)
        if scale:
            X = verify.scale(X)
            
        X = verify.covariates(X)

        self._verbose = verbose

        super(Generic, self).__init__(Y, X, Wmat, Mmat, Delta,
                                      n_samples=n_samples, n_jobs=n_jobs,
                                      extra_traced_params = extra_traced_params,
                                      priors=priors,
                                      starting_values=starting_values,
                                      configs=configs,
                                      truncation=truncation)
