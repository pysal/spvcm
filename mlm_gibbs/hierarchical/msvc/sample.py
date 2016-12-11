from __future__ import division
import scipy.stats as stats
import scipy.linalg as scla
import numpy as np
from warnings import warn as Warn
from ...utils import chol_mvn
from ...steps import metropolis

def mu_beta(state):
    """
    The sample step for mu_beta in a multi-process SVCP. Results in a draw from:

    N(Sn.dot(Mn), Sn),

    where Mn is Xt(Y - ZGammas - XcZetas)/Tau2 + S0^-1m0
    and Sn is (XtX/Tau2 + S0^-1)^-1

    modifies state.Mus and state.XMus inplace.
    """
    st = state

    Sni = st.XtX / st.Tau2 + st.mu_cov0i
    Sn = np.linalg.inv(Sni)
    Mn = (st.X.T.dot(st.Y - st.ZGammas -  st.XcZetas)) / st.Tau2
    Mn += (st.mu_cov0i.dot(st.mu_mean0))
    st.Mus = chol_mvn(Sn.dot(Mn), Sn)
    st.XMus = st.X.dot(st.Mus)
    return st.Mus

def tau(state):
    """
    The sample step for tau in a multi-process SVCP. Results in a draw from:

    IG(an,bn)
    where an is n/2+a0
    and bn is (Y-XMus-XcZetas - ZGammas)**2/2 + b0, where **2 is the matrix square

    modifies state.Tau2 inplace.
    """
    st = state
    st.eta = st.Y - st.XMus - st.XcZetas - st.ZGammas
    bn = st.eta.T.dot(st.eta) / 2 + st.tau_b0
    st.Tau2 = stats.invgamma.rvs(st.tau_an, scale=bn)
    return st.Tau2

def zeta(state):
    """


    modifies state.Zetas, state.XcZetas, and state.Zeta_list inplace
    """
    st = state
    zn = st.Xc.T.dot(st.Y - st.ZGammas - st.XMus) / st.Tau2
    Sigma_zni = st.XctXc / st.Tau2 + st.Hi
    Sigma_zn = np.linalg.inv(Sigma_zni)
    st.Zetas = chol_mvn(Sigma_zn.dot(zn), Sigma_zn)
    st.XcZetas = st.Xc.dot(st.Zetas)
    st.Zeta_list = np.split(st.Zetas, st.p)
    return st.Zetas

def gamma(state):
    """

    modifies st.Gammas and st.ZGammas in place, if the state has Z.
    """
    st = state
    eta_muzeta = st.Y - st.XMus - st.XcZetas
    gn = st.Z.T.dot(eta_muzeta) / st.Tau2
    gn += st.gammas_cov0i.dot(st.gammas_mean0)

    Sgni = st.ZtZ / st.Tau2 - st.gammas_cov0i
    Sgn = np.linalg.inv(Sgni)
    st.Gammas = chol_mvn(Sgn.dot(gn), Sgn)
    st.ZGammas = st.Z.dot(st.Gammas)
    return st.Gammas

def all_j(state):
    """
    Sample each process

    modifies Phi_list, H_list, Hi_list, Sigma2_list, H, Hi in place.
    """
    st = state
    for j in range(len(st.Phi_list)):
        new_phi_j = phi_j(st, j)
        st.Phi_list[j] = new_phi_j
        st.H_list[j] = st.correlation_function(new_phi_j, st.pwds)
        st.Hi_list[j] = np.linalg.inv(st.H_list[j])
        new_sigma_j = sigma_j(st, j)
        st.Sigma2_list[j] = new_sigma_j
    st.H = scla.block_diag(*[Hj * Sigma2j for Hj, Sigma2j in
                             zip(st.H_list, st.Sigma2_list)])
    st.Hi = scla.block_diag(*[Hji/Sigma2j for Hji, Sigma2j in
                              zip(st.Hi_list, st.Sigma2_list)])
    return st.Phi_list

def phi_j(state, idx):
    """
    The metropolis sample step for the `idx`th Phi value.

    modifies configurations only inplace.
    """
    state.j = idx
    st = state
    global_cfgs = state.configs
    this_cfg = global_cfgs.Phi[idx]
    current = state.Phi_list[idx]

    try:
        proposal = this_cfg.proposal
    except KeyError:
        proposal = stats.normal
        this_cfg.proposal = proposal

    new_val, accepted = metropolis(state, current, proposal,
                                   logp_phi_j, this_cfg.jump)
    if accepted:
        this_cfg.accepted += 1
    else:
        this_cfg.rejected += 1
    if global_cfgs.tuning:
        acc = this_cfg.accepted
        rej = this_cfg.rejected
        ar = acc / (acc + rej)
        if ar < this_cfg.ar_low:
            this_cfg.jump *= this_cfg.adapt_step
        elif ar > this_cfg.ar_hi:
            this_cfg.jump /= this_cfg.adapt_step
        if acc + rej > this_cfg.max_tuning:
            this_cfg.tuning = False
    return new_val

def sigma_j(state, idx):
    """
    The sample step for the `idx`th sigma2_j parameter.

    modifies nothing inplace.
    """
    st = state
    idx = st.j
    Hj = st.H_list[idx]
    Hji = st.Hi_list[idx] #could we get this out of Zeta's Hi?
    Zeta_j = st.Zeta_list[idx]
    Sigma2_j = st.Sigma2_list[idx]
    Phi_j = st.Phi_list[idx]
    a0_j = st.a0_list[idx]
    b0_j = st.b0_list[idx]

    bjn = Zeta_j.T.dot(Hji).dot(Zeta_j) / 2 + b0_j
    new_sigma = stats.invgamma.rvs(st.an_list[idx], scale=bjn)
    return new_sigma

def logp_phi_j(state, val, j=None):
    """
    The logp for the current phi parameter, where `j` is
    either stored in state or passed as an argument.

    modifies nothing inplace.
    """

    if j is None:
        idx = state.j
    else:
        idx = j
    if val < 0:
        return -np.inf
    st = state
    Hj = state.correlation_function(val, state.pwds)
    lu,p = scla.lu_factor(Hj)
    Zeta_j = st.Zeta_list[idx]
    Sigma2_j = st.Sigma2_list[idx]
    Phi_j = st.Phi_list[idx]
    c0_j = st.Phi_shape0_list[idx]
    d0_j = st.Phi_rate0_list[idx]


    logdet = -.5*np.sum(np.log(np.abs(lu.diagonal())))
    varterm = -(st.n / 2) * np.log(Sigma2_j)
    phiterm = (c0_j - 1) * np.log(Phi_j)
    kern = scla.lu_solve((lu,p), Zeta_j).T.dot(Zeta_j)
    norm_kern = (-1/(2*Sigma2_j) * kern)
    gamma_kern = - Phi_j * d0_j
    return logdet + varterm + norm_kern + gamma_kern
