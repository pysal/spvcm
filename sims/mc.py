import utils as u
from dgp import _mkDelta
import numpy as np
import samplers as samp

def gen_tests(df,W,M, outvar='Y', lowvar='X', upvar='Z'):
    parvals = [x.split('_')[1:] for x in df.columns if x.startswith('Y')]
    Wm = W.full()[0]
    Mm = M.full()[0]
    for r,l in parvals:
        yield (r,l,_mkrun(df,Wm,Mm,r,l, outvar, lowvar, upvar))

def _mkrun(df, W,M,r,l, outvar = 'Y', lowvar = 'X', upvar = 'Z'):
    """
    Setup a HSAR run from a dataframe, and parameters governing rho and lambda
    """
    Xvars = [x for x in df.columns if x.startswith('X')]
    Zvars = [z for z in df.columns if z.startswith('Z')]
    
    parstring = '_'.join([r,l])
    Design = df[Xvars + Zvars].values

    N,p = Design.shape
    J = M.shape[0]
    y = df['Y_'+parstring].values.reshape(N,1)
    X = Design
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
    rhospace = np.arange(-.99, .99,.001)
    rhospace = rhospace.reshape(rhospace.shape[0], 1)
    rhodets = np.array([u.LU_logdet(In - rho*W) for rho in rhospace]).reshape(rhospace.shape)
    rhos = np.hstack((rhospace, rhodets))
    lamspace = np.arange(-.99, .99, .001)
    lamspace = lamspace.reshape(lamspace.shape[0], 1)
    lamdets = np.array([u.LU_logdet(Ij - lam*M) for lam in lamspace]).reshape(lamspace.shape)
    lambdas = np.hstack((lamspace, lamdets))

    #invariants in rho sampling
    beta0 = u.lstsq(X, y).reshape(p, 1)
    e0 = y - np.dot(X, beta0)
    e0e0 = np.dot(e0.T, e0)

    Wy = np.dot(W, y)
    betad = u.lstsq(X, Wy).reshape(p,1)
    ed = Wy - np.dot(X, betad)
    eded = np.dot(ed.T, ed)
    e0ed = np.dot(e0.T, ed)

    ####Actual estimation, still troubleshooting here. 

    #mock a pymc3 trace
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
