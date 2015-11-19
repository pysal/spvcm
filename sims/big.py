import pysal as ps
import numpy as np
import dgp

W_lower = ps.lat2W(5000,5000) #underlying grid is 5000x5000
W_lower.transform = 'r'
W = W_lower.full()[0]
W_upper = ps.lat2W(50,50) #each regime is 100x100
W_upper.transform = 'r'
M = W_upper.full()[0]

N = W_lower.n
J = W_upper.n

X,Z = dgp.design_matrix(N,J)
Delta = dgp._mkDelta(N,J)
Y = dgp.outcome(X,Z,W,M,Delta, .62, .24) #setup problem with rho = .62, lam=.24
