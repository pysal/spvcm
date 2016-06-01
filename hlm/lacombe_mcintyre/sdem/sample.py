import numpy as np
import numpy.linalg as la
import scipy.linalg as scla
from pysal.spreg import spdot 
from ...steps import metropolis
from ...utils import splogdet

def sample(SDEM):
    st = SDEM.state

    ### Draw Sigma2, level 1 variance
    e1 = st.y - st.DeltaAlphas - st.XBetas

    d1 = np.dot(e1.T, e1) 
    d1 += st.Sigma2_prod # INVARIANT
    
    chirv = np.random.chisquare(st.N + st.Sigma2_v0, size=1)
    st.Sigma2 = d1 / chirv

    ### Draw Beta for level one covariates
    covm = la.inv(st.XtX / st.Sigma2 + st.Betas_cov0)
    resids = np.dot(st.X.T, (st.y - np.dot(st.Delta, st.Alphas))) / st.Sigma2
    resids += st.Betas_covm # INVARIANT
    mean = np.dot(covm, resids)
    st.Betas = chol_mvn(mean, covm)
    st.XBetas = np.dot(st.X, st.Betas)

    ### Draw Alphas, spatially-correlated random intercepts
    scaled_emp_covm = st.DeltatDelta/st.Sigma2 
    st.BtB = spdot(st.B.T, st.B)
    spatial_covm = st.BtB / st.Tau2
    covm = la.inv(scaled_emp_covm + spatial_covm)

    mean_kernel = np.dot(st.Delta.T, (st.y - np.dot(st.X, st.Betas))) / st.Sigma2
    BZGammas = spdot(st.B, np.dot(st.Z, st.Gammas))
    mean_spcorr = np.dot(st.BtB, BZGammas) / st.Tau2
    mean = np.dot(covm, mean_kernel + mean_spcorr)
    st.Alphas = chol_mvn(mean, covm)
    st.DeltaAlphas = np.dot(st.Delta, st.Alphas)

    ### Draw Tau, variance in spatially-varying intercept
    e2 = spdot(st.B, st.Alphas) - BZGammas
    d2 = np.dot(e2.T, e2)
    d2 += st.Tau2_prod # INVARIANT

    chirv = np.random.chisquare(st.N + st.Tau2_v0, size=1) 
    st.Tau2 = d2/chirv

    ### Draw Gammas, the upper-level effects
    covm = np.dot(st.Z.T, np.dot(st.BtB, st.Z))/st.Tau2 + st.Gammas_cov0
    covm = la.inv(covm)

    mean_kernel = np.dot(st.Z.T, np.dot(st.BtB, st.Alphas)) / st.Tau2 
    mean_kernel += st.Gammas_covm # INVARIANT
    mean = np.dot(covm, mean_kernel)
    st.Gammas = chol_mvn(mean, covm)
    st.BZGammas = spdot(st.B, np.dot(st.Z, st.Gammas))

    ### Draw Lambda, the upper-level spatial correlation in intercepts
    st.Lambda = sample_lambda(SDEM.configs.Lambda, st.Lambda, SDEM.state, 
                        logp = logp_lambda)
    st.B = st.Ij - st.Lambda * st.M

    SDEM.cycles += 1

#############################
# Analytical Sampling steps #
#############################

def chol_mvn(Mu, Sigma, overwrite_Sigma=True):
    """
    Sample from a Multivariate Normal given a mean & Covariance matrix, using
    cholesky decomposition of the covariance

    That is, new values are generated according to :
    New = Mu + cholesky(Sigma) . N(0,1)

    Parameters
    ----------
    Mu      :   np.ndarray (p,1)
                An array containing the means of the multivariate normal being
                sampled
    Sigma   :   np.ndarray (p,p)
                An array containing the covariance between the dimensions of the
                multivariate normal being sampled

    Returns
    -------
    np.ndarray of size (p,1) containing draws from the multivariate normal
    described by MVN(Mu, Sigma)
    """
    D = scla.cholesky(Sigma, overwrite_a = overwrite_Sigma)
    e = np.random.normal(0,1,size=Mu.shape)
    kernel = np.dot(D.T, e)
    return Mu + kernel

#############################
# SPATIAL SAMPLE METHODS    #
#############################

def sample_lambda(confs, value, state, logp):
    """
    Sample a spatial parameter according to the rules stored in the parameter's
    SDM.configs

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

def logp_lambda(state, val):
    """
    The logp for Lambda as defined in 
    """
    st = state
    
    #must truncate in logp otherwise sampling gets unstable
    if (val < st.Lambda_min) or (val > st.Lambda_max):
        return np.array([-np.inf])
    B = st.Ij - val * st.M
    logdet = splogdet(B)
    resids = np.dot(B, st.Alphas) - np.dot(B, np.dot(st.Z, st.Gammas))
    ssq = np.dot(resids.T, resids)
    return logdet + -.5 * ssq / st.Sigma2
