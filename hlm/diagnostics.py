import numpy as np
import pandas as pd
import warnings
from collections import OrderedDict
import copy

try:
    from rpy2 .rinterface import RRuntimeError
    from rpy2.robjects.packages import importr
    _coda = importr('coda')
    HAS_CODA = True
    HAS_RPY2 = True
except ImportError:
    HAS_CODA =  False
    HAS_RPY2 = False
except RRuntimeError:
    HAS_CODA = False
    HAS_RPY2 = True


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

######################
# Geweke Diagnostics #
######################

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

def _geweke_map(model = None, trace=None, chain=None,
           drop_frac=.1, hold_frac=.5, n_bins=50,
           varnames=None, variance_method='ar', **ar_kw):
    trace = _resolve_to_trace(model, trace, chain, varnames)
    stats = trace.map(_geweke_vector, drop=drop_frac, hold=hold_frac, n_bins=n_bins,
                      varfunc=_geweke_variance[variance_method])
    return stats

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

##################
# Effective Size #
##################

def effective_size(model=None, trace=None, chain=None, varnames=None,
                    use_R=False):
    """
    Compute the effective size of a trace, accounting for serial autocorrelation. 
    This statistic is:

    N * var(x)/spectral0(x)

    where spectral0(x) is the spectral density of x at lag 0, an
    autocorrelation-adjusted estimate of the sequence variance. 

    NOTE: the backend argument defaults to estimating the effective_size in
    python. But, the statsmodels.tsa.AR required for the spectral density
    estimate is *slow* for large chains. If you have a properly configured R
    installation with the python package `rpy2` and the R package `coda` installed,
    you can opt to pass through to CODA by passing `use_R=True`.
    
    Arguments 
    ----------
    Arguments
    ----------
    model   :   Any model object.
                must have an attached `trace` attribute. Takes precedence over
                `trace` and `chain` arguments.
    trace   :   abstracts.Trace
                a trace object containing data to compute the diagnostic
    chain   :   np.ndarray
                an array indexed by (m,n[,p]) containing m parallel runs of n samples
                of p covariates.
    varnames:   str or list of str
                set of variates to extract from the model or trace to to compute the 
                statistic. 
    use_R   :   bool (default: False)
                option to drop the computation of the effective size down to R's CODA 
                package. Requires: rpy2, working R installation, CODA R package
    """
    trace = _resolve_to_trace(model, trace, chain, varnames)
    stats = trace.map(_effective_size, use_R=use_R)
    return stats if len(stats) > 1 else stats[0]

def _effective_size(x, use_R=False):
    """
    Compute the effective size from a given flat array

    Arguments
    -----------
    x       :   np.ndarray
                flat vector of values to compute the effective sample size

    use_R   :   bool
                option to use rpy2+CODA or pure python implementation. If False, 
                the effective size computation may be unbearably slow on large data,
                due to slow AR fitting in statsmodels.
    """
    if use_R:
        if HAS_RPY2 and HAS_CODA:
            return _coda.effectiveSize(x)[0]
        elif HAS_RPY2 and not HAS_CODA:
            raise ImportError("No module named 'coda' in R.")
        else:
            raise ImportError("No module named 'rpy2'")
    else:
        spec = _spectrum0_ar(x)
        if spec == 0:
            return 0
        loss_factor = np.var(x, ddof=1)/spec
        return len(x) * loss_factor

#############################
# Highest Posterior Density #
#############################

def hpd_interval(model = None,  trace = None,  chain = None,  varnames = None,  p=.95):
    trace = _resolve_to_trace(model, trace, chain, varnames)
    stats = trace.map(_hpd_interval, p=p)
    return stats if len(stats) > 1 else stats[0]
    
def _hpd_interval(data, p=.95):
    data = np.sort(data)
    N = len(data)
    N_in = int(np.ceil(N*p))
    head = np.arange(0,N-N_in)
    tail = head+N_in
    pivot = np.argmin(data[tail] - data[head])
    return data[pivot], data[pivot+N_in]

#############
# Summarize #
#############

def summarize(trace, level=0):
    """
    Summarize a trace object, providing its mean, median, HPD, 
    standard deviation, and effective size.

    Arguments
    ---------
    trace   :   trace
                trace object on which to compute the summary
    level   :   int 
                ordered in terms of how much information reduction occurs. a level 0 summary 
                provides the output for each chain. A level 1 summary provides output 
                grouped over all chains. 
    """
    dfs = trace.to_df()
    if isinstance(dfs, list):
        multi_index = ['Chain_{}'.format(i) for i in range(len(dfs))]
        df = pd.concat(dfs, axis=1, keys=multi_index)
    else:
        df = pd.concat((dfs,), axis=1, keys=['Chain_0'])
    df = df.describe().T[['count', 'mean', '50%', 'std']]
    HPDs = hpd_interval(trace=trace, p=.95)
    if HAS_CODA:
        ESS = effective_size(trace=trace, use_R=True)
    else:
        warn('Computing effective sample size may take a while due to statsmodels.tsa.AR.'
                , stacklevel=2)
        ESS = effective_size(trace=trace, use_R=False)
    flattened_HPDs = []
    flattened_ESSs = []
    if isinstance(HPDs, dict):
        HPDs = [HPDs]
    if isinstance(ESS, dict):
        ESS = [ESS]
    for i_chain, chain in enumerate(HPDs):
        this_HPD = dict() 
        this_ESS = dict()
        for key,val in chain.items():
            if isinstance(val, list):
                for i, hpd_tuple in enumerate(val):
                    name = '{}_{}'.format(key, i)
                    this_HPD.update({name:hpd_tuple})
                    this_ESS.update({name:ESS[i_chain][key][i]})
            else:
                this_HPD.update({key:val})
                this_ESS.update({key:ESS[i_chain][key]})
        flattened_HPDs.append(this_HPD)
        flattened_ESSs.append(this_ESS)
    #return df, flattened_HPDs, flattened_ESSs
    df['HPD_low'] = None
    df['HPD_high'] = None
    df['N_effective'] = None
    for i, this_chain_HPD in enumerate(flattened_HPDs):
        this_chain_ESS = flattened_ESSs[i]
        outer_key = 'Chain_{}'.format(i)
        keys = [(outer_key, inner_key) for inner_key in this_chain_HPD.keys()]
        lows, highs = zip(*[this_chain_HPD[key[-1]] for key in keys])
        n_eff = [this_chain_ESS[key[-1]] for key in keys]
        df.ix[keys, 'HPD_low'] = lows
        df.ix[keys, 'HPD_high'] = highs
        df.ix[keys, 'N_effective'] = n_eff
    df['median'] = df['50%']
    df['N_iters'] = df['count'].apply(int)
    df['N_effective'] = df['N_effective'].apply(round)
    df.drop('count', axis=1, inplace=True)
    df['AR_loss'] = (df['N_iters'] - df['N_effective'])/df['N_iters']
    df = df[['mean', 'HPD_low', 'median', 'HPD_high', 'std', 'N_iters', 'N_effective', 'AR_loss']]
    if level>0:
        df = df.unstack()
        grand_mean = df['mean'].mean(axis=0)
        lowest_HPD = df['HPD_low'].min(axis=0)
        grand_median = df['median'].median(axis=0)
        highest_HPD = df['HPD_high'].max(axis=0)
        std = df['std'].mean(axis=0)
        neff = df['N_effective'].sum(axis=0)
        N = df['N_iters'].sum(axis=0)
        df = pd.concat([grand_mean, lowest_HPD, grand_median, 
                        highest_HPD, std, N, neff], axis=1)
        df.columns = ['grand_mean', 'min_HPD', 'grand_median', 'max_HPD', 'std', 
                      'sum(N_iters)', 'sum(N_effective)']
    return df

#############
# Utilities #
#############

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
            return trace.drop(*[var for var in trace.varnames
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
    as applied in CODA. Written to replicate R, so defaults change. 
    Note: this is very slow when there is a lot of data. 
    """
    try:
        from statsmodels.api import tsa
    except ImportError:
        raise ImportError('Statsmodels is required to use the AR(0) '
                           ' spectral density estimate of the variance.')
    if fit_kw == dict():
        fit_kw['ic']='aic'
        N = len(data) 
        # R uses the smaller of N-1 and 10*log10(N). We should replicate that. 
        maxlag = N-1 if N-1 <= 10*np.log10(N) else 10*np.log(N)
        fit_kw['maxlag'] = int(np.ceil(maxlag))
    ARM = tsa.AR(data, **spec_kw).fit(**fit_kw)
    alphas = ARM.params[1:]
    return ARM.sigma2 / (1 - alphas.sum())**2

_geweke_variance = dict()
_geweke_variance['ar'] = _spectrum0_ar
_geweke_variance['naive'] = _naive_var
