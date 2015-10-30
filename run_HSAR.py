import numpy as np
import time
from trace import Trace
import samplers as samp
from six import iteritems as diter
from numpy import linalg as la
import scipy as s
from scipy.linalg import solve
import pysal as ps
import pandas as pd

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
lamdets = np.array([la.slogdet(Ij - lam*M)[-1] for lam in lamspace]).reshape(lamspace.shape)
lambdas = np.hstack((lamspace, lamdets))

#invariants in rho sampling
beta0, resids, rank, svs = la.lstsq(xc, y)
e0 = y - np.dot(xc, beta0)
e0e0 = np.dot(e0, e0.T)

Wy = np.dot(W, y)
betad, resids, rank, svs = la.lstsq(xc, Wy)
ed = y - np.dot(xc, betad)
eded = np.dot(ed, ed.T)
e0ed = np.dot(e0, ed.T)

#mock a pymc3 trace
statics = globals()
stochastics = ['betas', 'thetas', 'sigma_e', 'sigma_u', 'rho', 'lam']
samplers = [samp.Betas, samp.Thetas, samp.Sigma_e, samp.Sigma_u, samp.Rho, samp.Lambda]
stochastics = {k:v for k,v in zip(stochastics, samplers)}
gSampler = samp.Gibbs(n=20, backend='trace.csv', statics=statics, **stochastics)

#trace = Trace(stochastics, 10, statics = globals(), )
#trace.update('betas', np.zeros((p,1)))
#trace.update('thetas', np.zeros((J, 1)))
#trace.update('sigma_e', 2)
#trace.update('sigma_u', 2)
#trace.update('rho', .5)
#trace.update('lam', .5)
#
#
#
#allsamp = time.time()
#for _ in range(9):
#    print('starting step')
#    onestep = time.time()
#    bt = time.time()
#    betas = samp.Betas(trace)
#    betas.sample()
#    print('\t beta took {}'.format(time.time() - bt))
#
#    thetat = time.time()
#    theta = samp.Thetas(trace)
#    theta.sample()
#    print('\t theta took {}'.format(time.time() - thetat))
#
#    sigma_et = time.time()
#    sigma_e = samp.Sigma_e(trace)
#    sigma_e.sample()
#    print('\t sigma_e took {}'.format(time.time() - sigma_et))
#
#    sigma_ut = time.time()
#    sigma_u = samp.Sigma_u(trace)
#    sigma_u.sample()
#    print('\t sigma_u took {}'.format(time.time() - sigma_ut))
#
#    rhot = time.time()
#    rho = samp.Rho(trace)
#    rho.sample()
#    print('\t dummy rho took {}'.format(time.time() - rhot))
#
#    lambdaht = time.time()
#    lambdah = samp.Lambda(trace)
#    lambdah.sample()
#    print('\t dummy lambda took {}'.format(time.time() - lambdaht))
#    print('whole step took {}'.format(time.time() - onestep))
#print('10 sampling steps took {}'.format(time.time() - allsamp))
