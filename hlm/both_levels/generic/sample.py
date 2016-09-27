import scipy.linalg as scla
import scipy.stats as stats
import scipy.sparse as spar
import numpy as np
import numpy.linalg as la
from ...utils import splogdet, chol_mvn
from ...steps import metropolis

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
                                   logp, confs.jump)
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
    
    eta = st.Y - st.XBetas - st.DeltaAlphas
    kernel = eta.T.dot(PsiRhoi).dot(eta) / st.Sigma2

    return -.5*logdet -.5 * kernel + st.Log_Rho0(val)

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

    return -.5*logdet - .5*kernel + st.Log_Lambda0(val)
