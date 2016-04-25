import numpy as np
from numpy import linalg as la
from scipy import stats
from warnings import warn as Warn
from six import iteritems as diter

from pysal.spreg.diagnostics import constant_check
from pysal.spreg.utils import spmultiply, sphstack, spmin, spmax, spdot

from ..abstracts import AbstractSampler
from ..utils import inversion_sample

__all__ = ['Betas', 'Thetas', 'Sigma_e', 'Sigma_u', 'SAC_Upper', 'SAC_Lower']

class Betas(AbstractSampler):
    """
    Sampler for the full beta conditional posterior in a HSAR model. 

    These are the covariate parameters in a Hierarchical Simultaneous
    Autoregressive Model
    """
    def __init__(self, state=None, initial=None, name=None):
        super(Betas, self).__init__(state=state, name=name) 
        if initial is None:
            try:
                p = self.state['p']
                initial = np.zeros((1,p))
            except TypeError:
                Warn('Initial default guess was not found in state')
        self.initial = initial

    def _cpost(self):
        """
        Full conditional posterior for Beta, as defined in Equation 26 of Dong
        & Harris (2014).
        """
        st = self.state._state #alias for shared state of sampler
        fr = self.state.front #alias for the most recent parameter values
        VV = st.XtX / fr.Sigma_e + st.invT0
        st.v_betas = la.inv(VV) #conditional posterior variance matrix
        st.A = np.asarray(st.In - fr.SAC_Lower * st.W)
        st.Ay = spdot(st.A, st.y)
        st.Delta_u = spdot(st.Delta, fr.Thetas) #recall, HSAR.R mislabels Delta->Z
        lprod = spdot(st.X.T, (st.Ay - st.Delta_u)/fr.Sigma_e)
        lprod += st.T0M0.reshape(st.p, 1)
        st.m_betas = spdot(st.v_betas, lprod) #conditional posterior mean
        new_betas = np.random.multivariate_normal(st.m_betas.flatten(), st.v_betas)
        st.Betas  = new_betas.reshape(fr.Betas.shape)
        return st.Betas

class Thetas(AbstractSampler):
    """
    Sampler for the full Theta conditional poster in an HSAR model

    These are the "upper-level" random effects for a Hierarchical Simulatenous
    Autoregressive Model
    """
    def __init__(self, state=None, initial=None, name=None):
        super(Thetas, self).__init__(state=state, name=name) 
        if initial is None:
            try:
                J = self.state['J']
                initial = np.zeros((J,1))
            except KeyError:
                Warn('Initial default guess was not found in state')
        self.initial = initial

    def _cpost(self):
        """
        Full conditional posterior for Theta, as defined in equation 28 in Dong
        & Harris (2014)
        """
        st = self.state._state
        fr = self.state.front
        st.B = np.asarray(st.Ij - fr.SAC_Upper * st.M) #upper-level laplacian
        tmpv_u = spdot(st.Delta.T, st.Delta)/fr.Sigma_e 
        tmpv_u += spdot(st.B.T, st.B)/fr.Sigma_u
        st.v_u = la.inv(tmpv_u) #conditional posterior variance matrix
        st.Xb = spdot(st.X, fr.Betas.T) #uses recent copy of betas
        lprod = spdot(st.Delta.T, st.Ay - st.Xb) / fr.Sigma_e
        st.m_u = spdot(st.v_u, lprod) #conditional posterior means
        new_u = np.random.multivariate_normal(st.m_u.flatten(), st.v_u)
        st.Thetas = new_u.reshape(fr.Thetas.shape)
        return st.Thetas

class Sigma_e(AbstractSampler):
    """
    Sampler for the full Sigma_e conditional posterior in an HSAR model

    This is the variance for the "lower-level" errors in a Hierarhical
    Simulatenous Autoregressive model. 
    """
    def __init__(self, state=None, initial=2, name=None):
        super(Sigma_e, self).__init__(state=state, name=name) 
        if initial is None:
            initial = 2
        self.initial = initial

    def _cpost(self):
        """
        Full conditional posterior for Sigma_e as defined in equation 30 in Dong
        & Harris (2014)
        """
        st = self.state._state
        e = st.Ay - st.Delta_u - st.Xb
        st.de = .5 * spdot(e.T, e) + st.d0
        st.Sigma_e = stats.invgamma.rvs(st.ce, scale=st.de)
        return st.Sigma_e

class Sigma_u(AbstractSampler):
    """
    Sampler for the full Sigma_u conditional posterior in an HSAR model

    This is the variance for the "upper-level" random effects in a Hierarchical
    Simulatenous Autoregressive model. 
    """
    def __init__(self, state=None, initial=2, name=None):
        super(Sigma_u, self).__init__(state=state, name=name) 
        if initial is None:
            initial = 2
        self.initial = initial

    def _cpost(self):
        """
        Full conditional posterior for Sigma_u as defined in equation 29 of
        Dong & Harris (2014)
        """
        fr = self.state.front
        st = self.state._state
        Bus = spdot(st.B, fr.Thetas)
        st.bu = .5 * spdot(Bus.T, Bus) + st.b0
        st.Sigma_u = stats.invgamma.rvs(st.au, scale=st.bu)
        return st.Sigma_u

class SAC_Upper(AbstractSampler):
    """
    Sampler for the full conditional of the spatially-aucorrelated 
    error parameter, called lambda.  

    This is the "upper-level" error spatial dependence coefficient in a Hierarchical
    Simultaneous Autoregressive model
    """
    def __init__(self, state=None, initial=.5, name=None):
        super(SAC_Upper, self).__init__(state=state, name=name) 
        if initial is None:
            initial = .5
        self.initial = initial

    def _cpost(self):
        """
        Will be the full conditional posterior distribution for sac_upper as defined
        in equation 32 of Dong & Harris (2014), but is currently Unif(-1,1). 
        """
        fr = self.state.front
        st = self.state._state
        parvals, logdets = st.SAC_Upper_grid
        iota = np.ones_like(parvals)
        
        uu = spdot(fr.Thetas.T, fr.Thetas)
        uMu = spdot(spdot(fr.Thetas.T, st.M), fr.Thetas)
        Mu = spdot(st.M, fr.Thetas)
        uMMu = spdot(Mu.T, Mu)

        st.S_sac_upper = uu*iota - 2*parvals*uMu + uMMu*parvals**2

        log_density = logdets - st.S_sac_upper/(2*fr.Sigma_u)
        log_density = log_density - log_density.max()

        st.density = np.exp(log_density)
        
        st.SAC_Upper = inversion_sample(st.density, grid=parvals)
        return st.SAC_Upper

class SAC_Lower(AbstractSampler):
    """
    Sampler for the full SAC_Lower conditional posterior of an HSAR model

    This is the "lower-level" response spatial dependence coefficient in a
    Hierarchical Simultaneous Autoregressive model
    """
    def __init__(self, state=None, initial=.5, name=None):
        super(SAC_Lower, self).__init__(state=state, name=name) 
        if initial is None:
            initial = .5
        self.initial = initial

    def _cpost(self):
        """
        Will be the full conditional posterior distribution for SAC_Lower as defined
        in equation 31 of Dong & Harris (2014), but is currently Unif(-1,1). 
        """
        fr = self.state.front
        st = self.state._state
        parvals, logdets = st.SAC_Lower_grid #values of sac_lower
        iota = np.ones_like(parvals)
       
       #have to compute eu-related parameters
        beta_u, resids, rank, svs = la.lstsq(st.X, st.Delta_u)
        eu = st.Delta_u - spdot(st.X, beta_u)
        eueu = spdot(eu.T, eu)
        e0eu = spdot(st.e0.T, eu)
        edeu = spdot(st.ed.T, eu)

        st.S_sac_lower = (st.e0e0*iota + parvals**2 * st.eded + eueu 
                - 2*parvals*st.e0ed - 2*e0eu + 2*parvals*edeu)
        
        log_density = logdets - st.S_sac_lower/(2. * fr.Sigma_e)
        adj = log_density.max()
        log_density = log_density - adj #downshift to zero?

        st.density = np.exp(log_density)

        st.SAC_Lower = inversion_sample(st.density, grid=parvals)
        return st.SAC_Lower

if __name__ == '__main__':
    samplers = []
    samplers.append(Betas())
    
