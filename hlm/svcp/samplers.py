from ..abstracts import AbstractSampler
from ..sputils import *
from scipy import stats

class Tau(AbstractSampler):
    def __init__(self):
        pass
    def _cpost(self):
        nu = st.hypers.a + st.n/2
        sb = st.hyper.b + .5 * (st.y - spdot(spdot(st.X, st.Betas).T,
                                             spdot(st.X, st.Betas)))
        st.Tau = stats.invgamma.rvs(nu, sb)
        return st.Tau

class Phi(AbstractSampler):
    def __init__(self):
        pass
    def _cpost(self):

    def _llike(self):

