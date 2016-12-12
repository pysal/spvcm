from pysal.spreg.utils import spdot
import scipy.linalg as scla
from scipy import sparse as spar
from scipy.sparse import linalg as spla
import numpy as np
from ...steps import metropolis
from ...utils import splogdet, spinv, speye_like, chol_mvn


def sample(SDM):
    """
    Step through the conditional posteriors and draw from them
    """
    st = SDM.state

    ### Sample the variance, Sigma
    st.e1 = st.Y - st.DeltaAlphas - st.XBetas
    st.d1 = spdot(st.e1.T, st.e1) + st.Sigma2_prod  #vo is initial nu,
                                                    # inital inverse chi-squared 
                                                    # dof parameter. 
    st.chi = np.random.chisquare(st.N+st.Sigma2_v0, size=1)
    st.Sigma2 = st.d1/st.chi


    ### Sample the first-level effects, Beta
    covm = spinv(st.XtX/st.Sigma2 + st.Betas_cov0)

    #this is invariant
    xyda = spdot(st.X.T, (st.Y - st.DeltaAlphas))
    mean = spdot(covm, xyda / st.Sigma2 + st.Betas_covm)
    st.Betas = chol_mvn(mean, covm)
    st.XBetas = spdot(st.X, st.Betas)

    ### Sample the pooled intercept, Alpha
    covm_kern = st.DeltatDelta / st.Sigma2
    covm_upper = spdot(st.B.T, st.B) / st.Tau2
    covm = spinv(covm_kern + covm_upper)
    mean_lower = spdot(st.Delta.T, st.Y - st.XBetas) / st.Sigma2
    mean_upper = spdot(st.B.T, st.ZGammas) / st.Tau2
    mean = spdot(covm, mean_lower + mean_upper)
    st.Alphas = chol_mvn(mean, covm)
    st.DeltaAlphas = spdot(st.Delta, st.Alphas)
   
    ### Draw Tau for upper-level variance
    e2 = spdot(st.B, st.Alphas) - st.ZGammas
    # tau2so * tau2vo is invariant
    d2 = spdot(e2.T, e2) + st.Tau2_prod
    chi = np.random.chisquare(st.J+st.Tau2_v0, size=1)
    st.Tau2 = d2/chi
    
    ### Draw gammas, level 2 covariates
    covm = spinv(st.ZtZ/st.Tau2 + st.Gammas_cov0)
    mean_kern = spdot(spdot(st.Z.T, st.B), st.Alphas)/st.Tau2 
    
    #this is invariant
    mean_kern += st.Gammas_covm
    mean = spdot(covm, mean_kern)
    st.Gammas = chol_mvn(mean, covm)
    st.ZGammas = spdot(st.Z, st.Gammas)


    ### draw Rho, the spatial dependence parameter
    st.Rho = sample_rho(SDM.configs.Rho, st.Rho, SDM.state, 
                        logp = logp_rho)
    st.B = spar.csc_matrix(st.Ij - st.Rho * st.M)
    SDM.cycles += 1

#############################
# SPATIAL SAMPLE METHODS    #
#############################

def sample_rho(confs, value, state, logp):
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

def logp_rho(state, val):
    """
    The logp for Rho.
    """
    st = state
    
    #must truncate in logp otherwise sampling gets unstable
    if (val < st.Rho_min) or (val > st.Rho_max):
        return np.array([-np.inf])
    B = spar.csc_matrix(st.Ij - val * st.M)
    logdet = splogdet(B)
    ssq = spdot(B, st.Alphas) - spdot(st.Z, st.Gammas)
    ssq = spdot(ssq.T, ssq) 
    return logdet + -.5  * ssq / st.Sigma2
