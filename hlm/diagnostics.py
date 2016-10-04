import numpy as np
from collections import OrderedDict
import copy


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


def geweke(model = None, trace=None, chain=None,
           drop_frac=.1, hold_frac=.5, n_bins=50,
           varnames=None, variance_method='ar', **ar_kw):
    """
    This computes the plotting version of Geweke's diagnostic for a given trace. The iterative version is due to Brooks. This implementation mirrors that in the `R` `coda` package.
    
    Arguments
    ----------
    model   :   Any model object.
                must have an attached `trace` attribute. Takes precedence over
                `trace` and `chain` arguments.
    trace   :   abstracts.Trace
                a trace object containing data to compute the diagnostic
    chain   :   np.ndarray
                an array indexed by (m,n[,p]) containing m parallel runs of n samples of p covariates.
    drop_frac:  float
                the number of observations to drop each step from the first (1-`keep_frac`)% of the chain
    keep_frac:  float
                the comparison group of observations used to compare the to
                the bins over the (1 - keep_frac)% of the chain
    n_bins  :   int
                number of bins to divide the first (1 - keep_frac)% of the chain
                into.
    varnames:   string or list of strings
                name or list of names of parameters to which the diagnostic should be applied.
    variance_method: str
                name of the variance method to be used. The default, `ar0`, uses the spectral density at lag 0, which is also used in CODA. This corrects for serial correlation in the variance estimate. The alternative, `naive`, is simply the sample variance.
    ar_kw   :   dict/keyword arguments
                If provided, must contain `spec_kw` and `fit_kw`. `spec_kw` is a dictionary of keyword arguments passed to the statsmodels AR class, and `fit_kw` is a dictionary of arguments passed to the subsequent AR.fit() call.
    """
    trace = _resolve_to_trace(model, trace, chain, varnames)
    if varnames is None:
        varnames = trace.varnames
    variance_function = _geweke_variance[variance_method]
    all_stats = []
    for i, chain in enumerate(trace.chains):
        all_stats.append(dict())
        for var in varnames:
            data = np.squeeze(trace[i,var])
            if data.ndim > 1:
                n,p = data.shape[0:2]
                rest = data.shape[2:0]
                if len(rest) == 0:
                    data = data.T
                elif len(rest) == 1:
                    data = data.reshape(n,p*rest[0]).T
                else:
                    raise Exception('Parameter "{}" shape not understood.'                  ' Please extract, shape it, and pass '
                                    ' as its own chain. '.format(var))
            else:
                data = data.reshape(1,-1)
            stats = [_geweke_vector(datum, drop_frac, hold_frac, n_bins=n_bins,
                                    varfunc=variance_function)
                    for datum in data]
            if len(stats) > 1:
                results = {"{}_{}".format(var, i):stat for i,stat in
                            enumerate(stats)}
            else:
                results = {var:stats[0]}
            all_stats[i].update(results)
    return all_stats

def _geweke_vector(data, drop, hold, n_bins, **kw):
    """
    Compute a vector of geweke statistics over the `data` vector. This proceeds like the `R` `CODA` package's geweke statistic. The first half of the data vector is split into `n_bins` segments. Then, the Geweke statistic is repeatedly computed over subsets of the data where a bin is dropped each step. This results in `n_bins` statistics.
    """
    in_play = (len(data)-1)//2
    to_drop = np.linspace(0, in_play, num=n_bins).astype(int)
    return np.squeeze([_geweke_statistic(data[drop_idx:], drop, hold, **kw)
                       for drop_idx in to_drop])
    
def _geweke_statistic(data, drop, hold, varfunc=None):
    """
    Compute a single geweke statistic, defining sets A, B:
    
    mean_A - mean_B / (var(A) + var(B))**.5
    
    where A is the first `drop` % of the `data` vector, B is the last `hold` % of the data vector.
    
    the variance function, `varfunc`, is the spectral density estimate of the variance.
    """
    if varfunc is None:
        varfunc = _spectrum0_ar
    hold_start = np.floor((len(data)-1) * hold).astype(int)
    bin_width = np.ceil((len(data)-1)*drop).astype(int)
    
    drop_data = data[:bin_width]
    hold_data = data[hold_start:]
    
    drop_mean = drop_data.mean()
    drop_var = varfunc(drop_data)
    n_drop = len(drop_data)
    
    hold_mean = hold_data.mean()
    hold_var = varfunc(hold_data)
    n_hold = len(hold_data)
    
    return ((drop_mean - hold_mean) / np.sqrt((drop_var / n_drop)
                                            +(hold_var / n_hold)))

def _resolve_to_trace(model, trace, chain, varnames):
    """
    Resolve a collection of information down to a trace. This reduces the
    passed arguments to a trace that can be used for analysis based on names
    in varnames.
    
    If `trace` is passed, it is subset according to `varnames`, and a copy returned. It takes precedence.
    Otherwise, if `model` is passed, its traces are taken.
    Finally, if `chain` is passed, a trace is constructed to structure the chain.
    In all cases, if `varnames` is passed, it is used to name or subset the given data.
    
    """
    n_passed = sum([model is not None, trace is not None, chain is not None])
    if n_passed > 1:
        raise Exception('Only one of `model`, `trace`, or `chain` '
                        ' may be passed.')
    if isinstance(varnames, str):
        varnames = [varnames]
    if trace is not None:
        if varnames is not None:
            return trace.drop([var for var in trace.varnames
                               if var not in varnames], inplace=False)
        else:
            return copy.deepcopy(trace)
    if model is not None:
        return _resolve_to_trace(model=None, trace=model.trace,
                                 chain=None, varnames=varnames)
    if chain is not None:
        m,n = chain.shape[0:2]
        rest = chain.shape[2:]
        new_p = np.multiply(*rest)
        chain = chain.reshape(m,n,new_p)
        if varnames is None:
            varnames = ['parameter_{}'.format(i) for i in new_p]
        else:
            if len(varnames) != new_p:
                raise NotImplementedError('Parameter Subsetting by varnames '
                                  'is not currenlty implented for raw arrays')
        return Trace([Hashmap({k:run.T[p] for p,k in enumerate(varnames)})
                      for run in chain])
        
def _naive_var(data, *_, **__):
    """
    Naive variance computation of a time `x`, ignoring dependence between the
    variance within different windows
    """
    return np.var(data, ddof=1)

def _spectrum0_ar(data, spec_kw=dict(), fit_kw=dict()):
    """
    The corrected spectral density estimate of time series variance,
    as applied in CODA
    """
    try:
        from statsmodels.api import tsa
    except ImportError:
        raise ImportError('Statsmodels is required to use the AR(0) '
                           ' spectral density estimate of the variance.')
    if fit_kw == dict():
        fit_kw['ic']='aic'
    ARM = tsa.AR(data, **spec_kw).fit(**fit_kw)
    alphas = ARM.params[1:]
    return ARM.sigma2 / (1 - alphas.sum())**2

            

_geweke_variance = dict()
_geweke_variance['ar'] = _spectrum0_ar
_geweke_variance['naive'] = _naive_var
