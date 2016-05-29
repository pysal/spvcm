import hlm
from hlm import HSAR
from hlm.dong_harris.verify import Delta_members
from hlm.utils import trace_to_df
import dgp
import pysal as ps
import pandas as pd
import numpy as np
import scipy.sparse as spar
from functools import partial
import itertools as it

PARAMETER_NAMES = ['Betas', 'Sigma2_e', 'Sigma2_u', 'Rho', 'Lambda']

W = ps.open('./test_data/w_lower.mtx').read()
M = ps.open('./test_data/w_upper.mtx').read()
df = pd.read_csv('./test_data/test.csv')
membership = df[['county']].values - 1
W.transform = M.transform = 'r'
Wmat = W.sparse
Mmat = M.sparse
Delta, mems = Delta_members(Delta = None, membership=membership, N=W.n, J=M.n)
W_min, W_max = hlm.utils.speigen_range(Wmat)
M_min, M_max = hlm.utils.speigen_range(Mmat)

rhogrid = np.arange(1./W_min, 1./W_max, .01)
lamgrid = np.arange(1./M_min, 1./M_max, .01)

In = np.identity(W.n)
Ij = np.identity(M.n)

As = [spar.csc_matrix(In - rho * Wmat) for rho in rhogrid]
Bs = [spar.csc_matrix(Ij - lam * Mmat) for lam in lamgrid]

rhodets = np.hstack([hlm.utils.splogdet(A) for A in As]).flatten()
lamdets = np.hstack([hlm.utils.splogdet(B) for B in Bs]).flatten()

rhos = np.vstack((rhogrid, rhodets))
lams = np.vstack((lamgrid, lamdets))

def experiment(**kw):
    Betas, Sigma2_e, Sigma2_u, Rho, Lambda = dgp.setup_params(**kw)
    config = 'se_{}-su_{}-rho_{}-lam_{}'
    config = config.format(Sigma2_e, Sigma2_u, Rho, Lambda)
    print('starting {}'.format(config))
    y, X = dgp.dgp(Betas, Sigma2_e, Sigma2_u, Rho, W, Lambda, M, Delta)
    expgrd =  HSAR(y,X,W,M,membership=membership, n_samples=5000,
                     spatial_method='grid', effects_method='chol',
                     rho_grid = rhos, lambda_grid=lams,
                     truncate=(W_min, W_max, M_min, M_max), verbose=3)
    expmet = HSAR(y,X,W,M,membership=membership, n_samples=5000,
                     spatial_method='met', effects_method='chol',
                     truncate=(W_min, W_max, M_min, M_max), verbose=3)
    ols = ps.spreg.OLS(y,X, W, spat_diag=True, moran=True)
    df = trace_to_df(expgrd.trace)
    df['method'] = 'grid'
    df.to_csv('./out/'+config+'-grid.csv', index=False)
    df = trace_to_df(expmet.trace)
    df['method'] = 'met'
    df.to_csv('./out/'+config+'-met.csv', index=False)
    with open('./out/' + config + '-ols.txt', 'w') as f:
        f.write(ols.summary)

def build_frame(**kw):
    """
    Build an experiment frame using the range or unique values 
    passed in as keywords. 

    Parameters
    ------------
    kw  :   keywords
            a name of a parameter in the HSAR and a tuple of (min,max,step) or a
            list of unique values to use for a grid of tests

    Returns
    ---------
    a generator that contsructs tests according to the cartesian product of the
    grid. That is, the generator will step through all of the parameter grids
    provided, like a nested for loop. So, if you passed:
    build_frame(Lambda=[-.5, 0, .5], Rho=[-.5, 0, .5], Sigma2_e = [.2, 2, 20]), 

    you would get experiments like:
    for lamb in Lambda:
        for rho in Rho:
            for sig2e in Sigma2_e:
                do_experiment(lamb, rho, sig2e)
    """
    frame = dict()
    for param in PARAMETER_NAMES:
        conf = kw.get(param, None)
        if conf is not None:
            #conf should be (min,max,step) tuple or array of values
            if isinstance(conf, (np.ndarray, list)):
                frame.update({param:conf})
            else:
                frame.update({param:np.arange(*conf)})
    allexps = it.product(*frame.values())
    labelled = [{k:v for k,v in zip(frame.keys(), exp)} 
                     for exp in allexps]
    return [partial(experiment, **l) for l in labelled]

def call_self(x):
    return x()
if __name__ == '__main__':
    import multiprocessing as mp
    
    Pool = mp.Pool(8)
    frame = build_frame(Lambda=[-.8, -.4, 0, .4, .8],
                        Rho=[-.8,-.4,0,.4,.8],
                        Sigma2_e=[.1,5])
    Pool.map(call_self, frame)
    Pool.close()
    import sys
    sys.exit()
