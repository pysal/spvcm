from __future__ import division
import scipy.stats as stats
import scipy.linalg as scla
import numpy as np
from warnings import warn as Warn
from ...utils import chol_mvn
from ...steps import metropolis

def sample(SVCP):
    """
    Conduct one iteration of a Gibbs sampler for the SVCP using the state
    provided. 
    """
    st = SVCP.state
    
    ## Tau, EQ 3 in appendix of Wheeler & Calder
    ## Inverse Gamma w/ update to scale, no change to dof
    y_Xbeta = st.Y - st.X.dot(st.Betas)
    scale = st.b0 + .5 * y_Xbeta.T.dot(y_Xbeta)
    st.Tau2 = stats.invgamma.rvs(st.Tau_dof, scale=scale)

    ##covariance: T, EQ 4 in appendix of Wheeler & Calder
    ## inverse wishart w/ update to covariance matrix, no change to dof
    st.H = st.correlation_function(st.Phi, st.pwds)
    st.Hinv = scla.inv(st.H)
    st.tiled_Hinv = np.linalg.multi_dot([st.np2n, st.Hinv, st.np2n.T])
    st.tiled_Mus = np.kron(st.iota_n, st.Mus.reshape(-1,1))
    st.info = (st.Betas - st.tiled_Mus).dot((st.Betas - st.tiled_Mus).T)
    st.kernel = np.multiply(st.tiled_Hinv, st.info) 
    st.covm_update = np.linalg.multi_dot([st.np2p.T, st.kernel, st.np2p])
    st.T = stats.invwishart.rvs(df=st.T_dof, scale=(st.covm_update + st.Omega0))

    ##mean hierarchical effects: mu_\beta, in EQ 5 of Wheeler & Calder
    ##normal with both a scale and a location update, priors don't change
    #compute scale of mu_\betas
    st.Sigma_beta = np.kron(st.H, st.T)
    st.Psi = np.linalg.multi_dot((st.X, st.Sigma_beta, st.X.T)) + st.Tau2 * st.In
    Psi_inv = scla.inv(st.Psi)
    S_notinv_update = np.linalg.multi_dot((st.Xs.T, Psi_inv, st.Xs))
    S = scla.inv(st.mu_cov0_inv + S_notinv_update)
    
    #compute location of mu_\betas
    mkernel_update = np.linalg.multi_dot((st.Xs.T, Psi_inv, st.Y))  
    st.Mu_means = np.dot(S, (mkernel_update + st.mu_kernel_prior))
    
    #draw them using cholesky decomposition: N(m, Sigma) = m + chol(Sigma).N(0,1)
    st.Mus = chol_mvn(st.Mu_means, S) 
    st.tiled_Mus = np.kron(st.iota_n, st.Mus)

    ##effects \beta, in equation 6 of Wheeler & Calder
    ##Normal with an update to both scale and location, priors don't change
    
    #compute scale of betas
    st.Tinv = scla.inv(st.T)
    st.kronHiTi = np.kron(st.Hinv, st.Tinv)
    Ai = st.XtX / st.Tau2 + st.kronHiTi
    A = scla.inv(Ai)
    
    #compute means of betas
    C = st.Xty / st.Tau2 + np.dot(st.kronHiTi, st.tiled_Mus)
    st.Beta_means = np.dot(A, C)
    st.Beta_cov = A
    
    #draw them using cholesky decomposition
    st.Betas = chol_mvn(st.Beta_means, st.Beta_cov)

    # local nonstationarity parameter Phi, in equation 7 in Wheeler & Calder
    # sample using metropolis
    sample_phi(SVCP)
    
def logp_phi(state, phi):
    """
    This is the log of the probability distribution in equation 7 of the
    appendix of the Wheeler and Calder (2010) paper on svcp
    """
    if phi < 0:
        return np.array([-np.inf])
    st = state
    # NOTE: I'm exploiting the following property for two square
    # matrices, H,T, of shapes n x n and p x p, respectively:
    # log(det(H kron T)) = log(det(H)^p * log(det(T)^n))
    # = log(det(H))*p + log(det(T))*n
    sgnH, logdetH = np.linalg.slogdet(st.H)
    sgnT, logdetT = np.linalg.slogdet(st.T)
    logdetH *= sgnH
    logdetT *= sgnT
    if any([x not in (-1,1) for x in [sgnH, sgnT]]):
        Warn('Catastrophic loss of precision in np.linalg.slogdet of np.kron(st.H, st.T)')
    logdet = logdetH * st.p + logdetT * st.n
    Bmu = st.Betas - st.tiled_Mus
    kronHT_inv = st.kronHiTi #since inv of kronecker is kronecker of invs
    normal_kernel = np.dot(Bmu.T, np.dot(kronHT_inv, Bmu)) * -.5
    gamma_kernel = np.log(phi)*(st.phi_shape0 - 1) + -1*st.phi_rate0*phi
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
    new_val, accepted = metropolis(state, current, proposal,
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
        if SVCP.cycles >= cfg.Phi.max_tuning:
            cfg.tuning = False

