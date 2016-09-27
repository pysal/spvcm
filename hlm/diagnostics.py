import numpy as np


#####################################
# Potential Scale Reduction Factors #
#####################################

def _gelman_rubin(chain):
    """
    Computes the original potential scale reduction factor from the 1992 paper by Gelman and Rubin:
    
    \sqrt{\frac{\hat{V}}{W} * \frac{dof}{dof-2}}
    
    where \hat{V} is a corrected estimate of the chain variance, composed of within- and between-chain variance components, W,B:
    
    V_hat = W*(1-1/n) + B/n + B/(n*m)
    
    and the degrees of freedom terms are:
    dof = 2 * V_hat**2 / (Var(W) + Var(B) + 2Cov(W,B)).
    
    The equations of the variance and covariance are drawn directly from the original paper. This implementation should come close to the implementation in the R CODA package, which computes the same normalization factor.
    
    If the chain is multivariate, it computes the statistic for each element of the multivariate chain.
    """
    m,n = chain.shape[0:2]
    rest = chain.shape[2:]
    if len(rest) == 0:
        chain_vars = chain.var(axis=1, ddof=1)
        chain_means = chain.mean(axis=1)
        grand_mean = chain.mean()

        W = np.mean(chain_vars)
        B = np.var(chain_means, ddof=1)*n
        sigma2_hat = W*(1-(1/n)) + B/n
        V_hat = sigma2_hat + B/(n*m)
        t_scale = np.sqrt(V_hat)

        #not sure if the chain.var(axis=1, ddof=1).var() is right.
        var_W = (1-(1/n))**2 * chain_vars.var(ddof=1) /  m
        var_B = ((m+1)/(m*n))**2 * (2*B**2) / (m-1)
        cov_s2xbar2 = np.cov(chain_vars, chain_means**2, ddof=1)[0,1]
        cov_s2xbarmu = 2 * grand_mean * np.cov(chain_vars, chain_means, ddof=1)[0,1]
        cov_WB = (m+1)*(n-1)/(m*n**2)*(m/n)*(cov_s2xbar2 - cov_s2xbarmu)

        t_dof = 2 * V_hat**2 / (var_W + var_B + 2* cov_WB)

        psrf = np.sqrt((V_hat / W) * t_dof/(t_dof - 2))
        
        return psrf
        
    else:
        return [_gelman_rubin(ch.T) for ch in chain.T]
        

def _brooks_gelman_rubin(chain):
    """
    Computes the Brooks and Gelman psrf in equation 1.1, pg. 437 of the Brooks and Gelman article.
    
    This form is:
    (n_chains + 1) / n_chains * (sigma2_hat / W) - (n-1)/(nm)
    where Sigma2_hat is the unbiased estimator for aggregate variance:
    (n-1)/n * W + B/n
    
    If the chain is multivariate, this computes the univariate version over all elements of the chain.
    """
    m,n = chain.shape[0:2]
    rest = chain.shape[2:]
    if len(rest) == 0:
        chain_vars = chain.var(axis=1, ddof=1)
        chain_means = chain.mean(axis=1)
        grand_mean = chain.mean()
        
        W = np.mean(chain_vars)
        B = np.var(chain_means, ddof=1)*n
        
        sigma2_hat = ((n-1)/n) * W + B/n
        Rhat = (m+1)/m * (sigma2_hat / W) - ((n-1)/(m*n))
        return np.sqrt(Rhat)
    else:
        return [_brooks_gelman_rubin(ch.T) for ch in chain.T]

_psrf = dict([('brooks', _brooks_gelman_rubin), ('original',_gelman_rubin)])

def psrf(model = None, trace=None, chain=None, autoburnin=True,
         varnames=None, method='brooks'):
    """
    Wrapper to compute the potential scale reduction factor for
    a trace object or an arbitrary chain from a MCMC sample.
    
    Arguments
    ----------
    trace       :  Trace
                   A trace object that contains more-than-one chain.
    chain       :  np.ndarray
                   An array with at least two dimensions where the indices are:
                   (m,n[, k]), where m is the number of traces and n is the number of iterations. If the parameter is k-variate, then the trailing dimension must be k.
    autoburnin  :  boolean
                   a flat denoting whether to automatically slice the chain at its midpoint, computing the psrf for only the second half.
    varnames    :  string or list of strings
                   collection of the names of variables to compute the psrf.
    method      :  string
                   the psrf statistic to be used. Recognized options:
                   - 'brooks' (default): the 1998 Brooks-Gelman-Rubin psrf
                   - 'original': the 1992 Gelman-Rubin psrf
    """
    if model is not None:
        trace = model.trace
    if trace is not None and varnames is None:
        varnames = trace.varnames
    elif chain is not None and varnames is None:
        varnames = ['parameter']
    elif chain is not None and varnames is not None:
        try:
            assert len(varnames) == 1
        except AssertionError:
            raise UserWarning('Multiple chains outside of a trace '
                              'are not currently supported')
    out = dict()
    for param in varnames:
        if chain is not None:
            this_chain = chain
            m,n = chain.shape[0:2]
            rest = chain.shape[2:]
        else:
            this_chain = trace[param]
            m,n = this_chain.shape[0:2]
            rest = this_chain.shape[2:]
        this_chain = this_chain[:,-n//2:,]
        out.update({param:_psrf[method](this_chain)})
    return out