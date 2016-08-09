import scipy.linalg as scla
import scipy.stats as stats
import scipy.sparse as spar
import numpy as np
import numpy.linalg as la
from ...utils import splogdet, chol_mvn
from ...steps import metropolis
from ...both_levels.generic.sample import logp_lambda, sample_spatial

def sample(Model):
    st = Model.state

    ### Sample the Beta conditional posterior
    ### P(beta | . ) \propto L(Y|.) \dot P(\beta) 
    ### is
    ### N(Sb, S) where
    ### S = (X' Sigma^{-1}_Y X + S_0^{-1})^{-1}
    ### b = X' Sigma^{-1}_Y (Y - Delta Alphas) + S^{-1}\mu_0
    covm_update = st.X.T.dot(st.X) / st.Sigma2
    covm_update += st.Betas_cov0i
    covm_update = la.inv(covm_update) 

    resids = st.y - st.Delta.dot(st.Alphas)
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

    resids = st.y - st.XBetas
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
    eta = st.y - st.XBetas - st.DeltaAlphas
    bn = eta.T.dot(eta) * .5 + st.Sigma2_b0
    st.Sigma2 = stats.invgamma.rvs(st.Sigma2_an, scale=bn)
        
    ### P(Psi(\rho) | . ) \propto L(Y | .) \dot P(\rho) 
    ### is 
    ### |Psi(rho)|^{-1/2} exp(1/2(eta'Psi(rho)^{-1}eta * Sigma2^{-1})) * 1/(emax-emin)
    st.Lambda = sample_spatial(Model.configs.Lambda, st.Lambda, st, 
                               logp=logp_lambda)
    st.PsiLambda = st.Psi_2(st.Lambda, st.M)
    st.PsiLambdai = la.inv(st.PsiLambda)
