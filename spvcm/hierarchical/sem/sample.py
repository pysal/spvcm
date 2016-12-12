import numpy as np
import numpy.linalg as la
import scipy.linalg as scla
import scipy.sparse as spar
import scipy.stats as stats
import numpy.linalg as la
from pysal.spreg import spdot 
from ...steps import metropolis
from ...utils import splogdet

def sample(SEM):
    st = SEM.state

    ### Draw Sigma2, level 1 variance
    
    eta = st.y - st.DeltaAlphas - st.XBetas

    bn = np.dot(eta.T, eta) / 2 
    bn += st.Sigma2_b0# INVARIANT

    st.Sigma2 = stats.invgamma.rvs(st.Sigma2_an, scale=bn)
    

    ### Draw Beta for level one covariates
    covm = la.inv(st.XtX / st.Sigma2 + st.Betas_cov0i)
    resids = st.y - np.dot(st.Delta, st.Alphas)
    data_part = np.dot(st.X.T, resids) / st.Sigma2
    full_update = data_part + st.Betas_covm # INVARIANT
    mean = np.dot(covm, full_update)
    st.Betas = chol_mvn(mean, covm)
    st.XBetas = np.dot(st.X, st.Betas)

    ### Draw Alphas, spatially-correlated random intercepts
    scaled_emp_covm = st.DeltatDelta/st.Sigma2 
    st.BtB = spdot(st.B.T, st.B)
    spatial_covm = st.BtB.dot(st.Ij / st.Tau2)
    covm = la.inv(scaled_emp_covm + spatial_covm)

    mean_kernel = np.dot(st.Delta.T, (st.y - st.XBetas)) / st.Sigma2
    mean = np.dot(covm, mean_kernel)
    st.Alphas = chol_mvn(mean, covm)
    st.DeltaAlphas = np.dot(st.Delta, st.Alphas)

    ### Draw Tau, variance in spatially-varying intercept
    bn = la.multi_dot([st.Alphas.T, st.BtB, st.Alphas]) / 2
    st.Tau2 = stats.invgamma.rvs(st.Tau2_an, scale=bn + st.Tau2_b0)


    ### Draw Lambda, the upper-level spatial correlation in intercepts
    st.Lambda = sample_lambda(SEM.configs.Lambda, st.Lambda, SEM.state, 
                        logp = logp_lambda)
    st.B = spar.csc_matrix(st.Ij - st.Lambda * st.M)

    SEM.cycles += 1

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
    BtB = B.T.dot(B)
    logdet = splogdet(B)
    resids = la.multi_dot([st.Alphas.T, BtB, st.Alphas]) 
    ssq = np.dot(resids.T, resids) / (2 * st.Sigma2) 
    return logdet - ssq + st.logLambda0 
