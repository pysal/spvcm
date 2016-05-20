import pandas as pd
import hlm
import pysal as ps
import numpy as np

np.random.seed(8879)
data = pd.read_csv('./test_data/test.csv')
y = data[['y']].values
X = data[['x']].values

W_low = ps.open('./test_data/w_lower.mtx').read()
W_low.transform = 'r'
W_up = ps.open('./test_data/w_upper.mtx').read()
W_up.transform = 'r'

membership = data[['county']].values

s = hlm.HSAR(y, X, W_low, W_up, membership=membership, n_samples=0)
s.sample(1000)

df = pd.DataFrame().from_records(s.trace.__dict__)
df.to_csv('./tracetest.csv')
