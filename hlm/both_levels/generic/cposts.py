import scipy.linalg as scla
import scipy.stats as stats
import scipy.sparse as spar
import numpy as np
import numpy.linalg as la
from ...utils import splogdet, chol_mvn
from ...steps import metropolis

try:
    from numba import autojit
except ImportError:
    def autojit(func):
        return func

@autojit
def betas(y, X, Delta, Alphas, PsiSigma2i, Betas_cov0i, Betas_mean0):
    covm_update = X.T.dot(PsiSigma2i).dot(X)
    covm_update += Betas_cov0i
    covm_update = la.inv(covm_update) 

    resids = y - Delta.dot(Alphas)
    XtSresids = X.T.dot(PsiSigma2i).dot(resids)
    mean_update = XtSresids + Betas_cov0i.dot(Betas_mean0)
    mean_update = np.dot(covm_update, mean_update)
    return chol_mvn(mean_update, covm_update)

@autojit
def alphas(y, XBetas, Delta, PsiSigma2i, PsiTau2i):
    covm_update = Delta.T.dot(PsiSigma2).dot(Delta)
    covm_update += PsiTau2i
    covm_update = la.inv(covm_update)

    resids = y - XBetas
    mean_update = Delta.T.dot(PsiSigma2i).dot(resids)
    mean_update = np.dot(covm_update, mean_update)
    return chol_mvn(mean_update, covm_update)

@autojit
def tau2(Alphas, PsiLambdai, Tau2_b0, Tau2_an):
    bn = Alphas.T.dot(PsiLambdai).dot(Alphas) * .5 + Tau2_b0
    return stats.invgamma.rvs(Tau2_an, scale=bn)

@autojit
def sigma2(y, XBetas, DeltaAlphas, Sigma2_b0):
    eta = y - XBetas - DeltaAlphas
    bn = eta.T.dot(PsiRhoi).dot(eta) * .5 + Sigma2_b0
    return stats.invgamma.rvs(Sigma2_an, scale=bn)


#############################
# SPATIAL SAMPLE METHODS    #
#############################

def sample_spatial(confs, value, state, logp):
    """
    Sample a spatial parameter according to the rules stored in the parameter's
    Generic.Parameter.configs

    Parameters
    ----------
    confs   :   Namespace
                a namespace containing the configuration options for the 
                parameter being sampled
    value   :   float or int
                the current value of the parameter
    logp    :   callable(state, value)
                a function that takes the state and a parameter value and
                returns the log of the probability density function

    Returns
    -------
    a new value of the spatial parameter, drawn according to the information in
    confs. 
    """
    new_val, accepted = metropolis(state, value, confs.proposal, 
                                   logp, confs)
    # increment relevant parameters
    if accepted:
        confs.accepted += 1
    else:
        confs.rejected += 1

    #adapt if in adaptive phase
    if confs.adapt:
        confs.ar = confs.accepted / (confs.rejected+confs.accepted)
        if confs.ar < confs.ar_low:
            confs.jump /= confs.adapt_step
        elif confs.ar > confs.ar_hi:
            confs.jump *= confs.adapt_step
    if (confs.accepted + confs.rejected) > confs.max_adapt:
            confs.adapt = False
    return new_val

def logp_rho(state, val):
    """
    The logp for lower-level spatial parameters in this case has the same
    form as a multivariate normal distribution, sampled over the variance matrix, rather than over y. 
    """
    st = state
    
    #must truncate in logp otherwise sampling gets unstable
    if (val < st.Rho_min) or (val > st.Rho_max):
        return np.array([-np.inf])
    
    PsiRho = st.Psi_1(val, st.W)
    PsiRhoi = la.inv(PsiRho)
    logdet = splogdet(PsiRho)
    
    eta = st.y - st.XBetas - st.DeltaAlphas
    kernel = eta.T.dot(PsiRhoi).dot(eta) / st.Sigma2

    return (-.5*logdet -.5 * kernel - (st.N/2)*np.log(np.pi*2*st.Sigma2) 
            + st.LogRho0)

def logp_lambda(state, val):
    """
    The logp for upper level spatial parameters in this case has the same form
    as a multivariate normal distribution, sampled over the variance matrix,
    rather than over Y.
    """
    st = state

    #must truncate
    if (val < st.Lambda_min) or (val > st.Lambda_max):
        return np.array([-np.inf])

    PsiLambda = st.Psi_2(val, st.M)
    PsiLambdai = la.inv(PsiLambda)

    logdet = splogdet(PsiLambda)

    kernel = st.Alphas.T.dot(PsiLambdai).dot(st.Alphas) / st.Tau2

    return (-.5*logdet - .5*kernel - (st.J/2)*np.log(np.pi*2*st.Tau2) 
            + st.LogLambda0)
