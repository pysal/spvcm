from __future__ import division
import scipy.stats as stats
import scipy.linalg as scla
import numpy as np
from warnings import warn as Warn
from util import nexp_weights

def H(phi, pwds, method=nexp_weights):
    """
    Compute the H matrix using a spatial dependence parameter, phi, a set of
    squareform pairwise distances, and a function that relates the two
    """
    return method(phi, pwds)


def sample(state):
    """
    Conduct one iteration of a Gibbs sampler for the SVCP using the state
    provided. 
    """
    st = state
    
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
    sample_phi(state)
    
    state._n_iterations += 1
    

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
    
def sample_phi(state):
    current = state.Phi
    state._phi_current_logp = logp_phi(state, state.Phi)
    try:
        #special proposals can be stored in configs
        proposal = state._configs['phi_proposal']
    except KeyError:
        #if we don't have a proposal, take it to be a normal proposal
        proposal = stats.normal
        # and short circuit this assignment for later
        state._configs['phi_proposal'] = proposal
    new_val, accepted, new_logp = metropolis(state, current,
                                   proposal, logp_phi,
                                   current_logp=state._phi_current_logp)

    state.Phi = new_val
    if accepted:
        state._configs['phi_accepted'] += 1
        state._phi_current_logp = new_logp
    else:
        state._configs['phi_rejected'] += 1

    if state._tuning:
        acc = state._configs['phi_accepted']
        rej = state._configs['phi_rejected']
        ar = acc / (acc + rej)
        if ar < .4:
            state._jump *= state._configs['phi_adapt_step']
        elif ar > .6:
            state._jump /= state._configs['phi_adapt_step']
        if state._n_iterations > state._max_tuning:
            state._tuning = False

def metropolis(state, current, proposal, logp, current_logp=None):
    new_val = proposal.rvs(loc=current, scale=state._jump)
    new_logp = logp(state, new_val)
    r = np.min((1, np.exp(new_logp - current_logp)))
    u = np.random.random()
    
    #print(u, r)
    #print(new_val, new_logp)
    #print(current, current_logp)

    if u < r:
        outval = new_val
        outlogp = new_logp
        accepted = True
    else:
        outval = current
        outlogp = current_logp
        accepted = False
    return outval, accepted, outlogp
