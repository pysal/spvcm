import scipy.linalg as scla
import scipy.stats as stats
import scipy.sparse as spar
import numpy as np
from ...utils import splogdet, chol_mvn
from ...steps import metropolis, inversion

def sample(HSAR):
    """
    Take one iteration of the Dong Harris HSAR sampler
    """
    st = HSAR.state
    
    ### Betas:
    ### Equation 27 of the Dong & Harris 2014
    Sigma_Betas = st.XtX / st.Sigma2_e + st.T0inv
    Sigma_Betas = scla.inv(Sigma_Betas) 
    
    st.A = np.asarray(st.In - st.Rho * st.W)
    st.Ay_DThetas = np.dot(st.A, st.y) - np.dot(st.Delta, st.Thetas)
    mean_update = np.dot(st.X.T, st.Ay_DThetas) / st.Sigma2_e
    Mu_Betas = np.dot(Sigma_Betas, mean_update + st.T0invM0)
    
    st.Betas = chol_mvn(Mu_Betas, Sigma_Betas)

    ### Thetas
    ### Equation 28 of the Dong & Harris 2014
    B = np.asarray(st.Ij - st.Lambda * st.M)
    st.BtB = np.dot(B.T, B)

    Sigma_Thetas = st.DtD/st.Sigma2_e + st.BtB / st.Sigma2_u
    Sigma_Thetas = scla.inv(Sigma_Thetas)

    st.Ay_Xbetas = np.dot(st.A, st.y) - np.dot(st.X, st.Betas)
    mean_kernel = np.dot(st.Delta.T, st.Ay_Xbetas) / st.Sigma2_e 
    Mu_Thetas = np.dot(Sigma_Thetas, mean_kernel)

    st.Thetas = chol_mvn(Mu_Thetas, Sigma_Thetas)

    ### Sigma2_u
    ### Equation 29 of Dong and Harris 2014
    ### Note: shape parameter is invariant, since J & a0 never change
    ### Note: DH parameterize the IG as Gelman (2003) do:
    ###         p(x) \propto x^{-a-1} \exp\{ -\beta / x\}
    TtBtBT = np.dot(st.Thetas.T, np.dot(st.BtB, st.Thetas))
    bu = TtBtBT/2. + st.b0

    st.Sigma2_u = stats.invgamma.rvs(st.au, scale=bu)

    ### Sigma2_e
    ### Equation 30 of Dong & Harris 2014
    ### Note: Shape/parameterization/invariants correspond to Sigma2_e
    deviations = st.Ay_Xbetas - np.dot(st.Delta, st.Thetas)
    ssds = np.dot(deviations.T, deviations)
    de = ssds*.5 + st.d0

    st.Sigma2_e = stats.invgamma.rvs(st.ce, scale=de)

    ### Rho
    ### Equation 31 of Dong & Harris 2014
    st.Rho = sample_spatial(HSAR.configs.Rho, st.Rho, st, 
                            logp=logp_rho, logp_kernel=logp_kernel_rho)

    ### Lambda
    ### Equation 33 of Dong & Harris 2014
    st.Lambda = sample_spatial(HSAR.configs.Lambda, st.Lambda, st, 
                               logp=logp_lambda, logp_kernel=logp_kernel_lambda)
    HSAR.cycles += 1

###########################
# Spatial Parameter Steps #
###########################

def sample_spatial(confs, value, state, logp = None, logp_kernel = None):
    """
    Sample a spatial autocorrelation parameter according to the
    rules stored in the parameter's HSAR.configs

    Parameters
    ----------
    confs       :   Namespace
                    a namespace containing the configurations for the parameter
                    being sampled.
    value       :   float or int
                    the current value of the parameter 
    state       :   Namespace
                    a namespace containing the current state contained in the
                    sampler
    logp        :   callable(state, value)
                    a function that takes the state and a parameter value and
                    returns the log of the probability density function,
                    conditional on the state
    logp_kernel :   callable(state, value)
                    a function that takes the state and a parameter value and
                    returns the kernel of the probability density function,
                    conditional on the state. The kernel should be all of the
                    terms that avoid evaluation of log determinants. 

    Returns
    --------
    A new value of the spatial parameter, drawn according to information in
    confs
    """
    if confs.sample_method is 'grid':
        # do inversion setup
        new_val = grid_sample(state, confs.grid, confs.logdets, logp_kernel)
    elif confs.sample_method.startswith('met'):
        # no setup for met needed
        new_val, accepted = metropolis(state, 
                                       value, 
                                       confs.proposal, 
                                       logp, 
                                       confs)
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

########################################
# Metropolis-Hastings Helper Functions #
########################################

def logp_rho(state, val):
    """
    The log probability density of the Rho parameter in equation 32
    of Dong & Harris 2014. 

    Its form is a spatially-filtered normal kernel:

    |I_n - \rho * W| \times 
            \exp\{-(2*\sigma_e^2)^{-1} 
                        (Ay - X\beta -\Delta\Theta)'(Ay - X\beta - \Delta\Theta)\}
    
    After logs:

    logdet(In - \rho * W) + (-2\sigma_e^2)^{-1}
                    \times  (Ay - X\beta - \Delta\Theta)'
                    \dot    (Ay - X\beta - \Delta\Theta)

    Thus, equation 32 is in error when it provides the log kernel as positive.
    
    Parameters
    ----------
    state       :   Namespace
                    collection of state of the sampler
    val         :   float/int
                    value of the parameter at which to compute the log pdf
    Returns
    -------
    value of the log probability density function evaluated at val, conditional
    on the current state of the sampler.
    """
    st = state
    #truncate, because logdet will dominate zero probability kernel
    if (val < st.Rho_min) or (val > st.Rho_max):
        return np.array([-np.inf])
    logdet = splogdet(spar.csc_matrix(st.In - val * st.W))
    kernel = logp_kernel_rho(state, val)
    return logdet + kernel

def logp_lambda(state, val):
    """
    The log probability density of the Lambda parameter in equation 33 of
    Dong & Harris 2014.

    Its form is a spatially-filtered normal kernel:

    |Ij - \lambda * M| \times \exp\{-(2\sigma_u^2)^{-1}(\Theta'B'B\Theta)\}

    After logs:

    logdet(Ij - \lambda * M) + (-2\sigma_u^2)^{-1} \times (\Theta'B'B\Theta)

    Thus, equation 34 is in error when it provides the second term as positive. 
    
    Parameters
    ----------
    state       :   Namespace
                    collection of state of the sampler
    val         :   float/int
                    value of the parameter at which to compute the log pdf
    
    Returns
    -------
    value of the log probability density function evaluated at val, conditional
    on the current state of the sampler.
    """
    st = state
    #truncate because logdet will dominante zero probability kernel
    if (val < st.Lambda_min) or (val > st.Lambda_max):
        return np.array([-np.inf])
    logdet = splogdet(spar.csc_matrix(st.Ij - val * st.M))
    kernel = logp_kernel_lambda(state, val)
    return logdet + kernel

#####################################
# Inversion Sample Helper Functions #
#####################################

def logp_kernel_rho(state, Rho):
    """
    This is the kernel of the logp in Dong & Harris 2014, equation 31

    As given, the density is:

    |I_n - \rho W| \times \exp\{-(2\sigma_e^2)^{-1}
                                (Ay - X\beta - \Delta\Theta)'
                                (Ay - X\beta - \Delta\Theta)\}

    Thus, the log is:

    logdet(I_n - \rho W) - (Ay - X\beta - \Delta\Theta)'
                           (Ay - X\beta - \Delta\Theta) / (2 \sigma_e^2)

    Then, the kernel is the second term:

    - (Ay - X\beta - \Delta\Theta)'(Ay - X\beta - \Delta\Theta) / (2 * \sigma_e^2)

    Thus, equation 32 is in error when it provides the log kernel as positive.
    """
    st = state
    A = np.asarray(st.In - Rho * st.W)
    ssds = np.dot(A, st.y) - np.dot(st.X, st.Betas) - np.dot(st.Delta, st.Thetas)
    return -1 * np.dot(ssds.T, ssds) / (2 * state.Sigma2_e)

def logp_kernel_lambda(state, Lambda):
    """
    This is the kernel of the logp in Dong & Harris 2014, equation 33

    As given, the density is:
    |I_j - \lambda M| \times \exp\{ -(2\sigma_u^2)^{-1} \Delta'B'B\Delta \}
    
    Thus, the log is:
    logdet(I_j - lambda M) - \Delta'B'B\Delta/(2\sigma_u^2)
    
    Then, the kernel is the second term, - \Delta'B'B\Delta / (2*\sigma_u)

    Thus, equation 34 is in error when it provides the log kernel as positive. 
    """
    st = state
    B = np.asarray(st.Ij - Lambda * st.M)
    TtBtBT = np.dot(np.dot(st.Thetas.T, B.T), np.dot(B,st.Thetas))
    return -1 * TtBtBT / (2. * st.Sigma2_u)
