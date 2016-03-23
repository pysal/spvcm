import utils as u
from dgp import _mkDelta
import numpy as np
import samplers as samp
import os
import shutil as sh

def mktests(df,W,M, outvar='Y', lowvar='X', upvar='Z', repickle='overwrite'):
    """
    Make experiment runs depending on the structure of the input dataframe. 

    Arguments
    =========
    df      :   dataframe
                dataframe where outcome variates are prefixed by outvar
    W       :   pysal.W
                spatial weights matrix for the lower level observations
    M       :   pysal.W
                spatial weights matrix for the upper level observations
    outvar  :   str
                name of the outcome variable in the experiment
    lowvar  :   str
                name of lower-level variates 
    upvar   :   str
                name of upper-level variates
    repickle:   str {'overwrite', 'read', 'skip'}
                an option flag denoting what to do with pickle behavior.
                'overwrite' will construct a new laplacian grid pickle for the
                rho and lambda parameters and use it in all subsequent tests to
                avoid recomputing the grid each time. 
                'read' will use the pickles in the current directory. 
                'skip' will avoid using pickles altogether.

    Returns
    ========
    one list of tuples of the form (rho, lambda, sampler), where rho and lambda
    are the values of the corresponding Y column of the dataframe and sampler is
    the gibbs sampler designed to estimate parameters using that response
    vector. 
    """
    parvals = [x.split('_')[1:] for x in df.columns if x.startswith(outvar)]
    Wm = W.full()[0]
    Mm = M.full()[0]
    results = []
    if repickle == overwrite:
       sh.rmtree('./.tmp_rhos.pkl')
       sh.rmtree('./.tmp_lams.pkl')
    for r,l in parvals:
        results.append((r,l,_mkrun(df,Wm,Mm,r,l, outvar, lowvar, upvar, repickle)))
        if repickle == 'overwrite': #only rewrite first time, read otherwise.
            repickle = 'read'
    return results

def _mkrun(df, W,M,r,l, outvar = 'Y', lowvar = 'X', upvar = 'Z', repickle='overwrite'):
    """
    Setup a HSAR run from a dataframe, and parameters governing rho and lambda

    Arguments
    ============
    df      :   dataframe
                a pandas dataframe containing the data with which to construct
                the experiment sampler
    W       :   numpy.ndarray
                the spatial weights matrix in array form for the lower-level
                observations. Can be sparse or dense
    M       :   numpy.ndarray
                the spatial weights matrix in array form for the upper-level
                observations. Can be sparse or dense
    r       :   float
                rho value to use for the simulation
    l       :   float 
                lambda value to use for the simulation
    outvar  :   str
                name of the outcome variable in the experiment
    lowvar  :   str
                name of lower-level variates 
    upvar   :   str
                name of upper-level variates
    pickle  :   str
                flag to determine whether to read in gridded gibbs array pickle
                from file or compute on the fly. 
    """
    Xvars = [x for x in df.columns if x.startswith(lowvar)]
    Zvars = [z for z in df.columns if z.startswith(upvar)]
    
    parstring = '_'.join([r,l])
    locovars = df[Xvars].values
    upcovars = df[Zvars].values
    N = locovars.shape[0]
    y = df['Y_'+parstring].values.reshape(N,1)

    X = np.hstack((np.ones_like(y), locovars))
    Z = np.hstack((np.ones_like(y), upcovars))
    X = np.hstack((X,Z))
    
    p = X.shape[1]
    J = M.shape[0]
    Delta = _mkDelta(N,J)
       
    ##Prior specs
    M0 = np.zeros(p)
    T0 = np.identity(p) * 100
    a0 = .01
    b0 = .01
    c0 = .01
    d0 = .01

    ##fixed matrix manipulations for MCMC loops
    XtX = np.dot(X.T, X)
    invT0 = u.invert(T0)
    T0M0 = np.dot(invT0, M0)

    ##unchanged posterior conditionals for sigma_e, sigma_u
    ce = N/2. + c0
    au = J/2. + a0
    
    In = np.identity(N)
    Ij = np.identity(J)
    ##set up griddy gibbs
    if repickle == 'read':
        rhos = np.load('.tmp_rhos.pkl')
        lambdas = np.load('.tmp_lams.pkl')
    else:
        rhospace = np.arange(-.99, .99,.001)
        rhospace = rhospace.reshape(rhospace.shape[0], 1)
        rhodets = np.array([u.LU_logdet(In - rho*W) for rho in rhospace]).reshape(rhospace.shape)
        rhos = np.hstack((rhospace, rhodets))
        lamspace = np.arange(-.99, .99, .001)
        lamspace = lamspace.reshape(lamspace.shape[0], 1)
        lamdets = np.array([u.LU_logdet(Ij - lam*M) for lam in lamspace]).reshape(lamspace.shape)
        lambdas = np.hstack((lamspace, lamdets))
        if repickle == overwrite:
            lamdets.dump('./tmp_lams.pkl')
            rhodets.dump('./tmp_rhodets.dump')
    #invariants in rho sampling
    beta0 = u.lstsq(X, y).reshape(p, 1)
    e0 = y - np.dot(X, beta0)
    e0e0 = np.dot(e0.T, e0)

    Wy = np.dot(W, y)
    betad = u.lstsq(X, Wy).reshape(p,1)
    ed = Wy - np.dot(X, betad)
    eded = np.dot(ed.T, ed)
    e0ed = np.dot(e0.T, ed)

    statics = locals()
    stochastics = ['betas', 'thetas', 'sigma_e', 'sigma_u', 'rho', 'lam']
    samplers = [samp.Betas, samp.Thetas, samp.Sigma_e, samp.Sigma_u, samp.Rho, samp.Lambda]
    gSampler = samp.Gibbs(*list(zip(stochastics, samplers)), n=1000, statics=statics)

    gSampler.trace.update('betas', np.zeros((1,p)))
    gSampler.trace.update('thetas', np.zeros((J,1)))
    gSampler.trace.update('sigma_e', 2)
    gSampler.trace.update('sigma_u', 2)
    gSampler.trace.update('rho', .5)
    gSampler.trace.update('lam', .5)
    gSampler.trace.pos += 1
    
    return gSampler
