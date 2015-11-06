import numpy as np
import samplers as samp
from six import iteritems as diter
from numpy import linalg as la
import pysal as ps
import pandas as pd
from rpy2.robjects import r as R

def setup_HSAR():
    data = pd.read_csv("./test.csv")
    y = data[['y']].values
    X = data[['x']].values

    xc = np.hstack((np.ones_like(y), X)) #data + constant

    W_low = ps.open('w_lower.mtx').read()
    W_low.transform = 'r'
    W_up = ps.open('w_upper.mtx').read()
    W_up.transform = 'r'

    #dimensions
    N = W_low.n #lower level units
    J = W_up.n #upper level units
    p = X.shape[-1]

    nj = data.groupby('county').count()[['x']].values
    Jid = data[['county']].values

    Delta = np.zeros((N, J))
    for county in data['county'].unique():
        Delta[data[data['county'] == county].index, county-1] = 1

    Zstar = np.ones(J).reshape((J,1))
    Z = np.dot(Delta, Zstar)

    #upper level dimension
    q = Z.shape[-1]

    #laplacian construction
    W = W_low.full()[0]
    M = W_up.full()[0]

    Weigs = la.eigvals(W)
    Meigs = la.eigvals(M)

    We_min, We_max = float(min(Weigs)), float(max(Weigs))
    Me_min, Me_max = float(min(Meigs)), float(max(Meigs))

    In = np.identity(N)
    Ij = np.identity(J)

    ##Prior specs
    M0 = np.zeros(p)
    T0 = np.identity(p) * 100
    a0 = .01
    b0 = .01
    c0 = .01
    d0 = .01

    ##fixed matrix manipulations for MCMC loops
    XtX = np.dot(X.T, X)
    invT0 = la.inv(T0)
    T0M0 = np.dot(invT0, M0)

    ##unchanged posterior conditionals for sigma_e, sigma_u
    ce = N/2. + c0
    au = J/2. + a0

    ##set up griddy gibbs
    rhospace = np.arange(We_min, We_max,.001)
    rhospace = rhospace.reshape(rhospace.shape[0], 1)
    rhodets = np.array([la.slogdet(In - rho*W) for rho in rhospace])
    rhodets = (rhodets[:,0] * rhodets[:,1]).reshape(rhospace.shape)
    rhos = np.hstack((rhospace, rhodets))
    lamspace = np.arange(Me_min, Me_max, .001)
    lamspace = lamspace.reshape(lamspace.shape[0], 1)
    lamdets = np.array([la.slogdet(Ij - lam*M)[-1] for lam in lamspace]).reshape(lamspace.shape)
    lambdas = np.hstack((lamspace, lamdets))

    #invariants in rho sampling
    beta0, resids, rank, svs = la.lstsq(xc, y)
    e0 = y - np.dot(xc, beta0)
    e0e0 = np.dot(e0.T, e0)

    Wy = np.dot(W, y)
    betad, resids, rank, svs = la.lstsq(xc, Wy)
    ed = y - np.dot(xc, betad)
    eded = np.dot(ed.T, ed)
    e0ed = np.dot(e0.T, ed)

    ####Actual estimation, still troubleshooting here. 

    #mock a pymc3 trace
    statics = locals()
    stochastics = ['betas', 'thetas', 'sigma_e', 'sigma_u', 'rho', 'lam']
    samplers = [samp.Betas, samp.Thetas, samp.Sigma_e, samp.Sigma_u, samp.Rho, samp.Lambda]
    gSampler = samp.Gibbs(*zip(stochastics, samplers), n=20, statics=statics)

    gSampler.trace.update('betas', np.zeros((1,p)))
    gSampler.trace.update('thetas', np.zeros((J,1)))
    gSampler.trace.update('sigma_e', 2)
    gSampler.trace.update('sigma_u', 2)
    gSampler.trace.update('rho', .5)
    gSampler.trace.update('lam', .5)
    return gSampler

if __name__ == '__main__':
    s = setup_HSAR()
    R("source('setup.R')")
    R("i <- 2")
