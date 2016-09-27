from __future__ import division

import types
import numpy as np
import scipy.stats as stats
import copy

from numpy import linalg as la
from warnings import warn as Warn
from ...abstracts import Sampler_Mixin, Hashmap, Trace
from ... import verify
from ...utils import speigen_range, splogdet, ind_covariance, chol_mvn
from ...steps import metropolis


SAMPLERS = ['Alphas', 'Betas', 'Sigma2', 'Tau2']

class Base_MVCM(Sampler_Mixin):
    """
    The class that actually ends up setting up the MVCM model. Sets configs,
    data, truncation, and initial parameters, and then attempts to apply the
    sample function n_samples times to the state.
    """
    def __init__(self, Y, X, Delta, n_samples=1000, **_configs):
        
        N, p = X.shape
        _N, J = Delta.shape
        n_jobs = _configs.pop('n_jobs', 1)
        self.state = Hashmap(**{'X':X, 'Y':Y, 'Delta':Delta,
                           'N':N, 'J':J, 'p':p })
        self.traced_params = SAMPLERS
        extras = _configs.pop('extra_tracked_params', None)
        if extras is not None:
            self.traced_params.extend(extra_tracked_params)
        hashmaps = [Hashmap(**{k:[] for k in self.traced_params})]*n_jobs
        self.trace = Trace(*hashmaps)
        leftovers = self._setup_data(**_configs)
        self._setup_configs(**leftovers)
        self._setup_initial_values()
        self._setup_covariance()
        
        self.cycles = 0
        

        try:
            self.sample(n_samples)
        except (np.linalg.LinAlgError, ValueError) as e:
            Warn('Encountered the following LinAlgError. '
                 'Model will return for debugging. \n {}'.format(e))


    def _setup_data(self, **hypers):
        In = np.identity(self.state.N)
        Ij = np.identity(self.state.J)
        ## Prior specifications
        Sigma2_a0 = hypers.pop('Sigma2_a0', .001)
        Sigma2_b0 = hypers.pop('Sigma2_b0', .001)
        Betas_cov0 = hypers.pop('Betas_cov0', np.eye(self.state.p) * 100)
        Betas_mean0 = hypers.pop('Betas_mean0', np.zeros((self.state.p, 1)))
        Tau2_a0 = hypers.pop('Tau2_a0', .001)
        Tau2_b0 = hypers.pop('Tau2_b0', .001)


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

    def _setup_configs(self, #would like to make these keyword only using *
                 #multi-parameter options
                 truncate='eigs', tuning=0,
                 #spatial parameter grid sample configurations:
                 Rho_jump=.5, Rho_ar_low=.4, Rho_ar_hi=.6,
                 Rho_proposal=stats.norm, Rho_adapt_step=1.01,
                 #spatial parameter grid sample configurations:
                 Lambda_jump=.5, Lambda_ar_low=.4, Lambda_ar_hi=.6,
                 Lambda_proposal=stats.norm, Lambda_adapt_step=1.01,
                 **kw):
        """
        Omnibus function to assign configuration parameters to the correct
        configuration namespace
        """
        self.configs = Hashmap()
        self.configs.Rho = Hashmap(jump=Rho_jump, ar_low=Rho_ar_low,
                                   ar_hi = Rho_ar_hi, proposal=Rho_proposal,
                                   adapt_step = Rho_adapt_step, rejected=0,
                                   accepted=0, max_adapt=tuning,
                                   adapt=tuning>0)
        self.configs.Lambda = Hashmap(jump=Lambda_jump, ar_low=Lambda_ar_low,
                                   ar_hi = Lambda_ar_hi, proposal=Lambda_proposal,
                                   adapt_step = Lambda_adapt_step, rejected=0,
                                   accepted=0, max_adapt=tuning,
                                   adapt=tuning>0)

    def _setup_initial_values(self):
        """
        Set abrbitrary starting values for the Metropolis sampler
        """
        Betas = np.zeros((self.state.p ,1))
        Alphas = np.zeros((self.state.J, 1))
        Sigma2 = 4
        Tau2 = 4
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
        st.Psi_1 = ind_covariance
        st.Psi_2 = ind_covariance
        
        st.PsiRho = st.In
        st.PsiLambda = st.Ij

        st.PsiSigma2 = st.In * st.Sigma2
        st.PsiSigma2i = la.inv(st.PsiSigma2)
        st.PsiTau2 = st.Ij * st.Tau2
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
        covm_update = st.X.T.dot(st.X) / st.Sigma2
        covm_update += st.Betas_cov0i
        covm_update = la.inv(covm_update)
    
        resids = st.Y - st.Delta.dot(st.Alphas)
        XtSresids = st.X.T.dot(resids) / st.Sigma2
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
        covm_update = st.Delta.T.dot(st.Delta) / st.Sigma2
        covm_update += st.PsiTau2i
        covm_update = la.inv(covm_update)
    
        resids = st.Y - st.XBetas
        mean_update = st.Delta.T.dot(resids) / st.Sigma2
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
        st.PsiTau2 = st.Ij * st.Tau2
        st.PsiTau2i = la.inv(st.PsiTau2)
        
        ### Sample the response aspatial variance parameter
        ### P(Sigma2 | . ) \propto L(Y | .) \dot P(Sigma2)
        ### is
        ### IG(N/2 + a0, eta'Psi(\rho)^{-1}eta * .5 + b0)
        ### Where eta is the linear predictor, Y - X\beta + \DeltaAlphas
        eta = st.Y - st.XBetas - st.DeltaAlphas
        bn = eta.T.dot(st.PsiRhoi).dot(eta) * .5 + st.Sigma2_b0
        st.Sigma2 = stats.invgamma.rvs(st.Sigma2_an, scale=bn)

class MVCM(Base_MVCM):
    """
    The class that intercepts & validates input
    """
    def __init__(self, Y, X, Z=None, Delta=None, membership=None,
                 #data options
                 sparse = True, transform ='r', n_samples=1000, verbose=False,
                 **options):

        N, _ = X.shape
        if Delta is not None:
            _,J = Delta.shape
        elif membership is not None:
            J = len(np.unique(membership))
        else:
            raise UserWarning("No Delta matrix nor membership classification provided. Refusing to arbitrarily assign units to upper-level regions.")
        Delta, membership = verify.Delta_members(Delta, membership, N, J)



        X = verify.covariates(X)

        self._verbose = verbose
        if Z is not None:
            Z = Delta.dot(Z)
            X = np.hstack((X,Z))
        super(MVCM, self).__init__(Y, X, Delta, n_samples,
                **options)
