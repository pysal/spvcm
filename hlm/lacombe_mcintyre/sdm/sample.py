from pysal.spreg.utils import spdot
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
    st.d1 = spdot(st.e1.T, st.e1) + st.Sigma2_prod  #vo is initial nu,
                                                    # inital inverse chi-squared 
                                                    # dof parameter. 
    st.chi = np.random.chisquare(n+st.Sigma2_v0, size=1)
    st.Sigma2 = st.d1/st.chi


    ### Sample the first-level effects, Beta
    covm = spinv(spdot(st.X.T, st.X)/st.Sigma2 + st.Betas_cov0)

    #this is invariant
    mean = Betas_covm
    xyda = spdot(st.X.T, (st.y - spdot(st.Delta, st.Alphas)))
    move = spdot(covm, xyda) / st.Sigma2
    mean += move
    st.Betas = chol_mvn(mean, covm, SDM.configs.Betas.overwrite)

    ### Sample the pooled intercept, Alpha
    covm_kern = spdot(st.Delta.T, st.Delta) / st.Sigma2
    covm_upper = spdot(spdot(st.B.T, st.I_J), B) / st.Tau2
    covm = spinv(covm_kern + covm_upper)
    mean_kern = spdot(st.Delta.T, st.y - spdot(st.X, st.Betas)) / st.Sigma2
    mean_upper= spdot(st.B.T, spdot(st.Z, st.Gammas)) / st.Tau2
    mean = spdot(covm, mean_kern + mean_hetske)
    st.Alphas = sample_mvn(mean, covm, SDM.configs.Alphas.overwrite)
   
    ### Draw Tau for upper-level variance
    e2 = spdot(st.B, st.Alphas) - spdot(st.Z, st.Gammas)
    # tau2so * tau2vo is invariant
    d2 = spdot(e2.T, e2) + st.Tau2_prod
    chi = np.random.chi(st.J+st.Tau2_v0, size=1)
    st.Tau2 = d2/chi
    
    ### Draw gammas, level 2 covariates
    covm = spinv(spdot(st.Z.T, st.Z)/st.Tau**2 + st.Gammas_cov0)
    mean_kern = spdot(spdot(st.Z.T, st.B), st.Alphas)/st.Tau2 
    
    #this is invariant
    mean_kern += st.Gammas_covm
    mean = spdot(covm, mean_kern)
    st.Gammas = sample_mvn(mean, covm, SDM.configs.Gammas.overwrite)


    ### draw Rho, the spatial dependence parameter
    st.Rho = sample_rho(SDM.configs.Rho, st.Rho, SDM.state, 
                        logp = logp_rho)

    SDM.cycles += 1

#############################
# ANALYTICAL SAMPLE METHODS #
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
