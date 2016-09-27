from __future__ import division
from warnings import warn
import numpy as np
import copy
import scipy.stats as stats
import scipy.sparse as spar
import numpy as np
import numpy.linalg as la

from ...steps import metropolis
from ...both_levels.generic.sample import logp_lambda, sample_spatial
from ...both_levels.generic import Base_Generic
from ...both_levels.generic.model import SAMPLERS as generic_parameters
from ... import verify
from ...utils import se_covariance, ind_covariance, splogdet, chol_mvn



SAMPLERS = ['Alphas', 'Betas', 'Sigma2', 'Tau2', 'Lambda']

class Base_Upper_SE(Base_Generic):
    """
    The class that actually ends up setting up the Generic model. Sets configs,
    data, truncation, and initial parameters, and then attempts to apply the
    sample function n_samples times to the state.
    """
    def __init__(self, Y, X, M, Delta, n_samples=1000, **_configs):
        W = np.eye((Delta.shape[0]))
        super(Base_Upper_SE, self).__init__(Y, X, W, M, Delta,
                                      n_samples=0, skip_covariance=True, **_configs)
        self.state.Psi_1 = ind_covariance
        self.state.Psi_2 = se_covariance
        self._setup_covariance()
        original_traced = copy.deepcopy(self.traced_params)
        to_drop = [k for k in original_traced if (k not in SAMPLERS and k in generic_parameters)]
        self.traced_params = copy.deepcopy(SAMPLERS)
        for param in to_drop:
            for i, _  in enumerate(self.trace.chains):
                del self.trace.chains[i][param]

        if n_samples > 0:
            try:
                self.sample(n_samples, n_jobs=n_jobs)
            except (np.linalg.LinAlgError, ValueError) as e:
                Warn('Encountered the following LinAlgError. '
                     'Model will return for debugging. \n {}'.format(e))

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
        covm_update += st.PsiLambdai / st.Tau2
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
        
        ### Sample the response aspatial variance parameter
        ### P(Sigma2 | . ) \propto L(Y | .) \dot P(Sigma2)
        ### is
        ### IG(N/2 + a0, eta'Psi(\rho)^{-1}eta * .5 + b0)
        ### Where eta is the linear predictor, Y - X\beta + \DeltaAlphas
        eta = st.Y - st.XBetas - st.DeltaAlphas
        bn = eta.T.dot(eta) * .5 + st.Sigma2_b0
        st.Sigma2 = stats.invgamma.rvs(st.Sigma2_an, scale=bn)
            
        ### P(Psi(\rho) | . ) \propto L(Y | .) \dot P(\rho)
        ### is
        ### |Psi(rho)|^{-1/2} exp(1/2(eta'Psi(rho)^{-1}eta * Sigma2^{-1})) * 1/(emax-emin)
        st.Lambda = sample_spatial(self.configs.Lambda, st.Lambda, st,
                                   logp=logp_lambda)
        st.PsiLambda = st.Psi_2(st.Lambda, st.M)
        st.PsiLambdai = la.inv(st.PsiLambda)

class Upper_SE(Base_Upper_SE):
    """
    The class that intercepts & validates input
    """
    def __init__(self, Y, X, M, Z=None, Delta=None, membership=None,
                 #data options
                 transform ='r', n_samples=1000, verbose=False,
                 **options):
        _, M = verify.weights(None, M, transform=transform)
        self.M = M
        Mmat = M.sparse

        N,_ = X.shape
        if Delta is not None:
            J = Delta.shape[1]
        elif membership is not None:
            J = len(np.unique(membership))

        Delta, membership = verify.Delta_members(Delta, membership, N, J)

        X = verify.covariates(X)

        self._verbose = verbose
        if Z is not None:
            Z = Delta.dot(Z)
            X = np.hstack((X,Z))
        super(Upper_SE, self).__init__(Y, X, Mmat, Delta, n_samples,
                **options)