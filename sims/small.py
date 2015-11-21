import pysal as ps
import numpy as np
import dgp
import pandas as pd
import mc
import multiprocessing as mp
from six import iteritems as diter

data, M,W = dgp.scenario(2,10)

ycols = [x for x in data.columns if x.startswith('Y')]
dcols = [x for x in data.columns if not x.startswith('Y')]
yclean = []

for y in ycols:
    col,r,l = y.split('_')
    if len(r) > 4:
        r = '0.0'
    if len(l) > 4:
        l = '0.0'
    yclean.append('_'.join([col,r,l]))

data.columns = yclean + dcols

tester = mc.gen_tests(data, W,M)

def fsample(x):
    r,l,s = x
    s.sample(1000)
    trx = np.vstack(s.trace.Stochastics['betas'])
    #trx = np.hstack((trx, np.hstack(s.trace.Stochastics['thetas']).T))
    trx = np.hstack((trx, 
                     np.vstack(s.trace.Stochastics['rho']),
                     np.vstack(s.trace.Stochastics['lam']),
                     np.vstack(s.trace.Stochastics['sigma_e']),
                     np.vstack(s.trace.Stochastics['sigma_u'])))
    #columns = ['beta_{}'.format(i) for i in range(len(s.trace.Stochastics['betas'][0]))]
    #columns += ['theta_{}'.format(i) for i in range(len(s.trace.Stochastics['thetas'][0].T))]
    #columns += ['rho'] + ['lambda'] + ['sigma_e'] + ['sigma_u']
    tdf = pd.DataFrame(trx)
    pstring = 'results/{}_{}'.format(r,l).replace('-', 'n')
    tdf.to_csv(pstring+'.csv'.format(r,l), index=False)
    return x

q = fsample(next(tester))

#P = mp.Pool(mp.cpu_count())
#P.map(fsample, tester)
