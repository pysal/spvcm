import pysal as ps
import numpy as np
import dgp

print("weights")
W_lower = ps.lat2W(10,10) #underlying grid is 100x100
W_lower.transform = 'r'
W = W_lower.full()[0]
W_upper = ps.lat2W(5,5) #each regime is 20x20
W_upper.transform = 'r'
M = W_upper.full()[0]

N = W_lower.n
J = W_upper.n

print("design")
X,Z = dgp.design_matrix(N,J)

print("delta")
Delta = dgp._mkDelta(N,J)

print("Y")
Y = dgp.outcome(X,Z,W,M,Delta,.64,.23)


