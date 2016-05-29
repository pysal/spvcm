from __future__ import division
import scipy.stats as stats
import scipy.linalg as scla
import numpy as np
from warnings import warn as Warn
from .utils import nexp_weights

def H(phi, pwds, method=nexp_weights):
    """
    Compute the H matrix using a spatial dependence parameter, phi, a set of
    squareform pairwise distances, and a function that relates the two
    """
    return method(phi, pwds)


def sample(SVCP):
    """
    Conduct one iteration of a Gibbs sampler for the SVCP using the state
    provided. 
    """
    st = SVCP.state
    
    ## Tau, EQ 3 in appendix of Wheeler & Calder
    ## Inverse Gamma w/ update to scale, no change to dof
    y_Xbeta = st.y - np.dot(st.X, st.Betas)
    scale = st.b0 + .5 * np.dot(y_Xbeta.T, y_Xbeta)
    st.Tau2 = stats.invgamma.rvs(st.tau_dof, scale=scale)

    ##covariance: T, EQ 4 in appendix of Wheeler & Calder
    ## inverse wishart w/ update to covariance matrix, no change to dof
    st.H = H(st.Phi, st.pwds)
    st.Hinv = scla.inv(st.H)
    beta_splits = np.split(st.Betas, st.n)
    covms = []
    for i, j in zip(*np.triu_indices_from(st.Hinv)):
        delts = np.dot(beta_splits[i] - st.Mus, (beta_splits[j] - st.Mus).T)
        covms.append(st.Hinv[i,j] * delts)
    covm_update = np.sum(covms, axis=0) + st.Omega0
    st.T = stats.invwishart.rvs(st.T_dof, covm_update)

    ##mean hierarchical effects: mu_\beta, in EQ 5 of Wheeler & Calder
    ##normal with both a scale and a location update, priors don't change
    
    #compute scale of mu_\betas
    st.Sigma_beta = np.kron(st.H, st.T)
    st.Psi = np.dot(np.dot(st.X, st.Sigma_beta), st.X.T) + st.Tau2 * st.In
    Psi_inv = scla.inv(st.Psi)
    S_notinv_update = np.dot(st.Xs.T, np.dot(Psi_inv, st.Xs))
    S = scla.inv(st.mu0_cov_inv + S_notinv_update)
    
    #compute location of mu_\betas
    mkernel_update = np.dot(st.Xs.T, np.dot(Psi_inv, st.y))  
    st.Mu_means = np.dot(S, (mkernel_update + st.mu_kernel_prior))
    
    #draw them using cholesky decomposition: N(m, Sigma) = m + chol(Sigma).N(0,1)
    D = scla.cholesky(S)
    e = np.random.normal(0,1, size=(st.p,1))
    st.Mus = st.Mu_means + np.dot(D, e) 
    

    ##effects \beta, in equation 6 of Wheeler & Calder
    ##Normal with an update to both scale and location, priors don't change
    
    #compute scale of betas
    st.Tinv = scla.inv(st.T)
    st.kronHiTi = np.kron(st.Hinv, st.Tinv)
    A_notinv = st.XtX / st.Tau2 + st.kronHiTi
    A = scla.inv(A_notinv)
    
    #compute means of betas
    st.tiled_Mus = np.kron(st.iota_n, st.Mus)
    C = st.Xty / st.Tau2 + np.dot(st.kronHiTi, st.tiled_Mus)
    st.Beta_means = np.dot(A, C)
    st.Beta_cov = A
    
    #draw them using cholesky decomposition
    D = scla.cholesky(A)
    e = np.random.normal(0,1, size=(st.n*st.p,1))
    st.Betas = st.Beta_means + np.dot(D, e)

    # local nonstationarity parameter Phi, in equation 7 in Wheeler & Calder
    # sample using metropolis
    sample_phi(SVCP)
    
    st._n_iterations += 1
    

def logp_phi(state, phi):
    """
    This is the log of the probability distribution in equation 7 of the
    appendix of the Wheeler and Calder (2010) paper on svcp
    """
    if phi < 0:
        return np.array([-np.inf])
    st = state
    sgn, logdet = np.linalg.slogdet(np.kron(st.H, st.T))
    if sgn not in (-1,1):
        Warn('Catastrophic loss of precision in np.linalg.slogdet of np.kron(st.H, st.T)')
    logdet *= sgn
    Bmu = st.Betas - st.tiled_Mus
    kronHT_inv = st.kronHiTi #since inv of kronecker is kronecker of invs
    normal_kernel = np.dot(Bmu.T, np.dot(kronHT_inv, Bmu)) * -.5
    gamma_kernel = np.log(phi)*(st.alpha0 - 1) + -1*st.lambda0*phi
    return -.5*logdet + normal_kernel + gamma_kernel
    
def sample_phi(SVCP):
    """
    Sample phi, conditional on the state contained in the SVCP sampler

    Parameters
    ----------
    SVCP    :   sampler
                the execution context in which phi is to be sampled

    Returns
    --------
    None. works by sampling in place on SVCP. It updates:
    configs.phi.accepted OR configs.phi.rejected
    configs.phi.jump if tuning
    configs.phi.tuning if ending tuning
    state.Phi
    """
    state = SVCP.state
    cfg = SVCP.configs
    current = state.Phi
    try:
        #special proposals can be stored in configs
        proposal = cfg.Phi.proposal
    except KeyError:
        #if we don't have a proposal, take it to be a normal proposal
        proposal = stats.normal
        # and short circuit this assignment for later
        cfg.Phi.proposal = proposal
    new_val, accepted, new_logp = metropolis(state, current, proposal,
                                             logp_phi,cfg.Phi.jump)

    state.Phi = new_val
    if accepted:
        cfg.Phi.accepted += 1
    else:
        cfg.Phi.rejected += 1
    if cfg.tuning:
        acc = cfg.Phi.accepted
        rej = cfg.Phi.rejected
        ar = acc / (acc + rej)
        if ar < cfg.Phi.ar_low:
            cfg.Phi.jump *= cfg.Phi.adapt_step
        elif ar > cfg.Phi.ar_hi:
            cfg.Phi.jump /= cfg.Phi.adapt_step
        if state._n_iterations > cfg.max_tuning:
            cfg.tuning = False

def metropolis(state, current, proposal, logp, jump):
    """
    Simple metropolis algorithm for symmetric proposals.

    Parameters
    ----------
    state       :   Namespace
                    the current state of the sampler in a dict-like container
    current     :   numeric
                    the current value of the parameter being sampled
    proposal    :   scipy distribution
                    a distribution that supports a `.rvs(loc=, scale=)` method
    logp        :   callable(state, value)
                    a function that computes the log pdf of the value,
                    conditional on the information in state
    jump        :   numeric
                    the current value of the proposal scale parameter
    """
    current_logp = logp(state, current)
    new_val = proposal.rvs(loc=current, scale=jump)
    new_logp = logp(state, new_val)
    r = np.min((1, np.exp(new_logp - current_logp)))
    u = np.random.random()
    
    if u < r:
        outval = new_val
        outlogp = new_logp
        accepted = True
    else:
        outval = current
        outlogp = current_logp
        accepted = False
    return outval, accepted, outlogp
