import scipy.linalg as scla
import scipy.stats as stats
import scipy.sparse as spar
import numpy as np
import numpy.linalg as la
from ...utils import splogdet, chol_mvn
from ...steps import metropolis

def sample(Model):
    st = Model.state

    ### Sample the Beta conditional posterior
    ### P(beta | . ) \propto L(Y|.) \dot P(\beta) 
    ### is
    ### N(Sb, S) where
    ### S = (X' Sigma^{-1}_Y X + S_0^{-1})^{-1}
    ### b = X' Sigma^{-1}_Y (Y - Delta Alphas) + S^{-1}\mu_0
    covm_update = st.X.T.dot(st.PsiSigma2i).dot(st.X)
    covm_update += st.Betas_cov0i
    covm_update = la.inv(covm_update) 

    resids = st.y - st.Delta.dot(st.Alphas)
    XtSresids = st.X.T.dot(st.PsiSigma2i).dot(resids)
    mean_update = XtSresids + st.Betas_cov0i.dot(st.Betas_mean0)
    mean_update = np.dot(covm_update, mean_update)
    st.Betas = chol_mvn(mean_update, covm_update)
    st.XBetas = np.dot(st.X, st.Betas)

    ### Sample the Random Effect conditional posterior
    ### P( Alpha | . ) \propto L(Y|.) \dot P(Alpha | \lambda, Tau2)
    ###                               \dot P(Tau2) \dot P(\lambda)
    ### is
    ### N(Sb, S)
    ### Where
    ### S = (Delta'Sigma_Y^{-1}Delta + Sigma_Alpha^{-1})^{-1}
    ### b = (Delta'Sigma_Y^{-1}(Y - X\beta) + 0)
    covm_update = st.Delta.T.dot(st.PsiSigma2).dot(st.Delta)
    covm_update += st.PsiTau2i
    covm_update = la.inv(covm_update)

    resids = st.y - st.XBetas
    mean_update = st.Delta.T.dot(st.PsiSigma2i).dot(resids)
    mean_update = np.dot(covm_update, mean_update)
    st.Alphas = chol_mvn(mean_update, covm_update)
    st.DeltaAlphas = np.dot(st.Delta, st.Alphas)

    ### Sample the Random Effect aspatial variance parameter
    ### P(Tau2 | .) \propto L(Y|.) \dot P(\Alpha | \lambda, Tau2)
    ###                            \dot P(Tau2) \dot P(\lambda)
    ### is
    ### IG(J/2 + a0, u'(\Psi(\lambda))^{-1}u * .5 + b0)
    bn = st.Alphas.T.dot(st.PsiLambdai).dot(st.Alphas) * .5 + st.Tau2_b0
    st.Tau2 = stats.invgamma.rvs(st.Tau2_an, scale=bn)
    
    ### Sample the response aspatial variance parameter
    ### P(Sigma2 | . ) \propto L(Y | .) \dot P(Sigma2)
    ### is
    ### IG(N/2 + a0, eta'Psi(\rho)^{-1}eta * .5 + b0)
    ### Where eta is the linear predictor, Y - X\beta + \DeltaAlphas
    eta = st.y - st.XBetas - st.DeltaAlphas
    bn = eta.T.dot(st.PsiRhoi).dot(eta) * .5 + st.Sigma2_b0
    st.Sigma2 = stats.invgamma.rvs(st.Sigma2_an, scale=bn)

    ### Sample the spatial components using metropolis-hastings
    ### P(Psi(\lambda) | .) \propto L(Y | .) \dot P(\lambda) 
    ### is
    ### |Psi(lambda)|^{-1/2} exp(1/2(Alphas'Psi(lambda)^{-1}Alphas * Tau2^{-1}))
    ###  * 1/(emax-emin)
    st.Rho = sample_spatial(Model.configs.Rho, st.Rho, st, 
                            logp=logp_rho)
    
    st.PsiRho = st.Psi_1(st.Rho, st.W)
    st.PsiSigma2 = st.PsiRho * st.Sigma2
    st.PsiSigma2i = la.inv(st.PsiSigma2)
    st.PsiRhoi = la.inv(st.PsiRho)
        
    ### P(Psi(\rho) | . ) \propto L(Y | .) \dot P(\rho) 
    ### is 
    ### |Psi(rho)|^{-1/2} exp(1/2(eta'Psi(rho)^{-1}eta * Sigma2^{-1})) * 1/(emax-emin)
    st.Lambda = sample_spatial(Model.configs.Lambda, st.Lambda, st, 
                               logp=logp_lambda)
    st.PsiLambda = st.Psi_2(st.Lambda, st.M)
    st.PsiTau2 = st.PsiLambda * st.Tau2
    st.PsiTau2i = la.inv(st.PsiTau2)
    st.PsiLambdai = la.inv(st.PsiLambda)

    Model.cycles += 1

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
    
    eta = st.y - st.XBetas - st.DeltaAlphas
    kernel = eta.T.dot(PsiRhoi).dot(eta) / st.Sigma2

    return (-.5*logdet -.5 * kernel - (st.N/2)*np.log(np.pi*2*st.Sigma2) 
            + st.LogRho0)

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

    return (-.5*logdet - .5*kernel - (st.J/2)*np.log(np.pi*2*st.Tau2) 
            + st.LogLambda0)
