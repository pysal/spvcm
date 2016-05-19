import scipy.linalg as scla
import scipy.stats as stats
import numpy as np
from ..utils import splogdet
from steps import metropolis, inversion

#X is nxp
#Beta is px1
#theta is jx1
#Delta is nxj

def sample(HSAR):
    """
    Take one iteration of the Dong Harris HSAR sampler
    """
    st = HSAR.state

    ### Betas:
    ### Equation 27 of the Dong & Harris 2014
    Sigma_Betas = st.XtX / st.Sigma2_e + st.T0inv
    Sigma_Betas = scla.inv(Sigma_Beta) 
    
    st.A = np.asarray(st.In - st.Rho * st.W)
    st.Ay_Xbetas = np.dot(st.A, st.y) - np.dot(st.Delta, st.Thetas)
    mean_update = np.dot(st.X.T, st.Ay_Xbetas) / st.Sigma2_e
    Mu_Betas = np.dot(Sigma_Betas, mean_update + st.T0invM0)
    
    st.Betas = chol_mvn(Mu_Betas, Sigma_Betas)

    ### Thetas
    ### Equation 28 of the Dong & Harris 2014
    B = np.asarray(st.In - st.Lambda * st.M)
    st.BtB = np.dot(B.T, B)

    Sigma_Thetas = st.DtD/st.Sigma2_e + st.BtB / st.Sigma2_u
    Sigma_Thetas = scla.inv(Sigma_Thetas)

    mean_kernel = np.dot(st.Delta.T, st.Ay_Xbetas) / st.Sigma2_e 
    Mu_Thetas = np.dot(Sigma_Thetas, mean_kernel)

    st.Thetas = chol_mvn(Mu_Thetas, Sigma_Thetas)

    ### Sigma2_u
    ### Equation 29 of Dong and Harris 2014
    ### Note: shape parameter is invariant, since J & a0 never change
    ### Note: DH parameterize the IG as Gelman (2003) do:
    ###         p(x) \propto x^{-a-1} \exp\{ -\beta / x\}
    st.DtBtBD = np.dot(st.Thetas.T, np.dot(st.BtB, st.Thetas))
    bu = scale_update = st.b0

    st.Sigma2_u = stats.invgamma.rvs(st.au, scale=bu)

    ### Sigma2_e
    ### Equation 30 of Dong & Harris 2014
    ### Note: Shape/parameterization/invariants correspond to Sigma2_e
    st.deviations = st.Ay_Xbetas - np.dot(Delta, st.Thetas)
    st.ssds = np.dot(deviations.T, deviations)
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
    Sample a lower-level spatial autocorrelation parameter according to the
    rules stored in HSAR.configs.rho

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
        new_val = grid_sample(confs.grid, confs.logdets, logp_kernel_rho)
    elif confs.sample_method.startswith('met'):
        # no setup for met needed
        new_val, accepted = metropolis(state, 
                                       value, 
                                       confs.proposal, 
                                       logp, 
                                       confs)
        # increment relevant parameters
        if accepted:
            confs.n_accepted += 1
        else:
            confs.n_rejected += 1

        #adapt if in adaptive phase
        if confs.adapt:
            confs.ar = confs.acccepted / (confs.n_rejected+confs.n_accepted)
            if confs.ar < confs.target_low:
                confs.jump /= confs.adapt_rate
            elif confs.ar > confs.target_hi:
                confs.jump *= confs.adapt_rate
        if (confs.n_accepted + confs.n_rejected) > confs.max_adapt:
            confs.adapt = False
    return new_val

########################################
# Metropolis-Hastings Helper Functions #
########################################

def logp_rho(state, val=state.Rho):
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
    if (val < 1./st.W_emin) or (val > st.W_emax):
        return np.array([0])
    logdet = splogdet(st.In - val * st.W)
    kernel = logp_kernel_rho(state, val)
    return logdet + kernel

def logp_lambda(state, val=state.Lambda):
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
    if (val < 1./st.M_emin) or (val > 1/st.M_emax):
        return np.array([0])
    logdet = splogdet(st.Ij - val * st.M)
    kernel = logp_kernel_lambda(state, val)
    return logdet + kernel

##################################################
# Analytically-Known Conditional Posterior Draws #
##################################################

def sample_mvn(Mu, Sigma, confs):
    """
    Draw a multivariate normal sample according to the configurations provided.

    Parameters
    ----------
    Mu      :   np.ndarray (p,1)
                A vector of means for the multivariate normal distribution that is
                being sampled
    Sigma   :   np.ndarray (p,1)
                An array describing the covariance between dimensions of the
                multivariate normal distribution that is being sampled
    confs   :   Namespace
                a namespace that describes the configuration options for the
                parameter being sampled
    """
    try:
        p = Sigma.shape[0]
        assert Mu.shape[0] == p
        assert Mu.shape[1] == 1
    except AssertionError:
            raise ValueError('Provided Mu array is not of shape (p,1) and is not'
                              'conformal with the Sigma array provided. ')

    if confs.sample_method.startswith('m'):
        return numpy_mvn(Mu, Sigma)
    else:
        return chol_mvn(Mu, Sigma, overwrite_Sigma=confs.get('overwrite', True))


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
    kernel = np.dot(D, e)
    return Mu + kernel

def numpy_mvn(Mu, Sigma):
    """
    Sample from a Multivariate Normal given a mean and a covariance matrix,
    using the numpy.random.multivariate_normal function. 

    That is, new values are generated directly according to:
    N(Mu, Sigma)

    Parameters
    ----------
    Mu      :   np.ndarray (p,1)
                an array containing the means of the multivariate normal being
                sampled
    Sigma   :   np.ndarray (p,p)
                An array containing the covariance between dimensions of the
                multivariate normal being sampled

    Returns
    -------
    np.ndarray of size (p,1) containing draws from the multivariate normal
    described as MVN(Mu, Sigma)
    """
    return np.random.multivariate_normal(Mu.flatten(), Sigma).reshape(Mu.shape)

#####################################
# Inversion Sample Helper Functions #
#####################################

def grid_sample(state, X, logdets, logp_kernel):
    """
    This uses the kernel of a log probability distribution, a support, and a log
    determinant at each support point and returns a value from X according to
    the total logp
    
    Parameters
    ----------
    X       :   np.ndarray (k,)
                A flat array containing all k grid values over which the sample
                is drawn
    logdets :   np.ndarray (k,)
                A flat array containing the log determinant of a matrix at all k
                grid values over which the sample is drawn.
    logp_kernel :   callable(val
    """
    st = state

    ### Note: equation 34/36 are in error when suggesting the kernel is positive
    ###       It is, in fact, negative. So, the kernel function returns negative
    ###       kernels that are added to the log determinants to get the correct logp
    kernels = np.asarray([logp_kernel(state, x) for x in X]).flatten()
    
    log_density = logdets + kernels
    log_density -= log_density.max() #just in case there's a float overflow

    density = np.exp(log_density)
    new_val = inversion(density, grid=X)
    return new_val

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
    return -1 * ssds / (2 * state.Sigma2_e)

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
    B = np.asarray(st.Ij - Lambda * state.M)
    DtBtBD = np.dot(np.dot(state.Delta.T, B.T), np.dot(B,state.Delta))
    return -1 * DtBtBD / (2 * state.Sigma2_u)
