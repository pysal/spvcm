from pysal.spreg.sputils import spdot
from hlm.utils import spinv, speye_like
import scipy.linalg as scla
from scipy import sparse as spar
from scipy.sparse import linalg as spla


def sample(SDM):
    """
    Step through the conditional posteriors and draw from them
    """
    st = SDM.state

    ### Sample the variance, Sigma
    st.e1 = st.Y - spdot(st.Delta, st.Alphas) - spdot(st.X,  st.Betas)
    st.d1 = spdot(st.e1.T, st.e1) + sig2so * sig2vo #vo is initial nu,
                                                    # inital inverse chi-squared 
                                                    # dof parameter. 
    st.chi = np.random.chisquare(n+sig2vo, size=1)
    st.Sigma = np.sqrt(st.d1/st.chi)


    ### Sample the first-level effects, Beta
    covm = spinv(spdot(st.X.T, st.X)/st.Sigma**2 + betacovprior)

    #this is invariant
    mean = spdot(betacovprior, betameanprior)
    xyda = spdot(st.X.T, (st.y - spdot(st.Delta,st.Alpha)))
    move = spdot(covm, xyda) / st.Sigma**2
    mean += move
    zs = np.random.normal(0, 1, size=st.p).reshape(st.p, 1)
    st.Betas = mean + spdot(scla.cholesky(covm).T,zs)
    
    ### Sample the pooled intercept, Alpha
    covm_kern = spdot(st.Delta.T, st.Delta) / st.Sigma**2
    covm_upper = spdot(spdot(st.B.T, st.I_J), B) / st.Tau**2
    covm = spinv(covm_kern + covm_upper)
    mean_kern = spdot(st.Delta.T, st.y - spdot(st.X, st.Betas)) / st.Sigma**2
    mean_upper= spdot(st.B.T, spdot(st.Z, st.Gammas)) / st.Tau**2
    mean = spdot(covm, mean_kern + mean_hetske)
    zs = np.random.normal(0,1,size=(st.J,1))
    st.Alphas = mean + spdot(scla.choleksy(covm).T, zs)
   
    ### Draw Tau for upper-level variance
    e2 = spdot(st.B, st.Alphas) - spdot(st.Z, st.Gammas)
    # tau2so * tau2vo is invariant
    d2 = spdot(e2.T, e2) + tau2so * tau2vo
    chi = np.random.chi(st.J+tau2vo, size=1)
    st.Tau = np.sqrt(d2/chi)
    
    ### Draw gammas, level 2 covariates
    covm = spinv(spdot(st.Z.T, st.Z)/st.Tau**2 + st.gammacovprior)
    mean_kern = spdot(spdot(st.Z.T, st.B), st.Alphas)/st.Tau**2 
    
    #this is invariant
    mean_kern += gammacovprior * gammameanprior
    mean = spdot(covm, mean_kern)
    zs = np.random.normal(0,1,size=st.q).reshape(st.q,1)
    st.Gammas = mean + spdot(scla.chol(covm).T, zs)
    
    ### draw Rho, the spatial dependence parameter
    st.Rho = sample_rho(SDM.configs.Rho, st.Rho, SDM.state, 
                        logp = logp_rho)

    SDM.cycles += 1
##########################
# SPATIAL SAMPLE METHODS #
##########################

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
    The logp for Rho as defined in 
    """
    st = state
    
    #must truncate in logp otherwise sampling gets unstable
    if (val < st.Rho_min) or (val > st.Rho_max):
        return np.array([-np.inf])
    logdet = splogdet(val)
    ssq = spdot(val, st.Alphas) - spdot(st.Z, st.Gammas)
    ssq = spdot(ssq.T, ssq) 
    return -.5 * st.Sigma**2 * ssq + st.rhoprior
