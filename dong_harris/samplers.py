import numpy as np
from numpy import linalg as la
from scipy import stats
from warnings import warn as Warn
from six import iteritems as diter

from pysal.spreg.diagnostics import constant_check
from pysal.spreg.utils import spmultiply, sphstack, spmin, spmax, spdot

from ..samplers import AbstractSampler

__all__ = ['Betas', 'Thetas', 'Sigma_e', 'Sigma_u', 'SAC_Upper', 'SAC_Lower']

class Betas(AbstractSampler):
    """
    Sampler for the full beta conditional posterior in a HSAR model. 

    These are the covariate parameters in a Hierarchical Simultaneous
    Autoregressive Model
    """
    def __init__(self, state=None, initial=None):
        self.requires = ['XtX', 'In', 'W', 'invT0', 'y', 'X', 'T0M0', 'Delta', 'p']
        self.exports = ['Ay', 'A', 'v_betas', 'm_betas']
        self.state = state
        if initial is None:
            try:
                p = self.state['p']
                initial = np.zeros((p,1))
            except TypeError:
                Warn('Initial default guess was not found in state')
        self.initial = initial

    def _cpost(self, state=None):
        """
        Full conditional posterior for Beta, as defined in Equation 26 of Dong
        & Harris (2014).
        """
        if state is None:
            state = self.state
        for name in self.requires:
            try:
                exec("{n} = state['{n}']".format(n=name))
            except KeyError as KError:
                err = "Variable {} not found in state".format(name)
                err += " at step {}".format(self.__class__)
                raise KeyError(err)
            except TypeError:
                err = "No state found for sampler. Please set the state by running sampler.state = globals() or equivalent."
                raise TypeError(err)
        pt = self.state.front #grab most current sampled values
        VV = XtX / pt['Sigma_e'] + invT0
        v_betas = la.inv(VV) #conditional posterior variance matrix
        A = In - pt['SAC_Lower'] * W
        Ay = np.dot(A, y)
        Delta_u = np.dot(Delta, pt['Thetas']) #recall, HSAR.R labels Delta from paper as Z
        lprod = np.dot(X.T, (Ay - Delta_u)/pt['Sigma_e']) + T0M0.reshape(p,1)
        m_betas = np.dot(v_betas, lprod) #conditional posterior mean
        new_betas = np.random.multivariate_normal(m_betas.flatten(), v_betas)
        new_betas = new_betas.reshape(pt['Betas'].shape)
        self.state['Betas'] = new_betas #update in place
        for name in self.exports:
            state[name] = eval(name)
        return state

class Thetas(AbstractSampler):
    """
    Sampler for the full Theta conditional poster in an HSAR model

    These are the "upper-level" random effects for a Hierarchical Simulatenous
    Autoregressive Model
    """
    def __init__(self, state=None, initial=None):
        self.required = ['Ij', 'M', 'X', 'y', 'J', 'Ay', 'Delta']
        self.exports = ['Xb', 'B', 'm_u', 'v_u']
        self.state = state
        if initial is None:
            try:
                J = self.state['J']
                initial = np.zeros((J,1))
            except KeyError:
                Warn('Initial default guess was not found in state')
        self.initial = initial

    def _cpost(self, state=None):
        """
        Full conditional posterior for Theta, as defined in equation 28 in Dong
        & Harris (2014)
        """
        if state is None:
            state = self.state
        for name in self.requires:
            try:
                exec("{n} = state['{n}']".format(n=name))
            except KeyError:
                err = "Variable {} not found in trace".format(name)
                err += " at step {}".format(self.__class__)
                raise KeyError(err)
        pt = self.state.front
        B = Ij - pt['SAC_Upper'] * M #upper-level laplacian
        v_u = np.dot(Delta.T, Delta)/pt['Sigma_e'] + np.dot(B.T, B)/pt['Sigma_u']
        v_u = la.inv(v_u) #conditional posterior variance matrix
        Xb = np.dot(X, pt['Betas'].T) #uses recent copy of betas
        lprod = np.dot(Delta.T, Ay - Xb) / pt['Sigma_e']
        m_u = np.dot(v_u, lprod) #conditional posterior means
        new_u = np.random.multivariate_normal(m_u.flatten(), v_u)
        new_u = new_u.reshape(pt['Thetas'].shape)
        self.state['Thetas'] = new_u
        for name in self.exports:
            self.state[name] = eval(name)
        return state

class Sigma_e(AbstractSampler):
    """
    Sampler for the full Sigma_e conditional posterior in an HSAR model

    This is the variance for the "lower-level" errors in a Hierarhical
    Simulatenous Autoregressive model. 
    """
    def __init__(self, state=None, initial=2):
        self.requires = ['Delta', 'Ay', 'Xb', 'ce', 'd0']
        self.exports = ['Delta_u', 'de']
        self.state = state
        if initial is None:
            initial = 2
        self.initial = initial

    def _cpost(self, state=None):
        """
        Full conditional posterior for Sigma_e as defined in equation 30 in Dong
        & Harris (2014)
        """
        if state is None:
            state = self.state
        pt = self.trace.front()
        for name in self.requires:
            try:
                exec("{n} = state['{n}']".format(n=name))
            except KeyError:
                err = "Variable {} not found in trace".format(name)
                err += " at step {}".format(self.__class__)
                raise KeyError(err)
        pt = self.state.front
        Delta_u = np.dot(Delta, pt['Thetas'])
        e = Ay - Delta_u - Xb
        de = .5 * np.dot(e.T, e) + d0
        new_sigma_e = stats.invgamma.rvs(ce, scale=de)
        self.state['Sigma_e'] = new_sigma_e
        for name in self.exports:
            self.trace.Derived[name] = eval(name)
        return state

class Sigma_u(AbstractSampler):
    """
    Sampler for the full Sigma_u conditional posterior in an HSAR model

    This is the variance for the "upper-level" random effects in a Hierarchical
    Simulatenous Autoregressive model. 
    """
    def __init__(self, state=None, initial=2):
        self.requires = ['B', 'b0', 'au']
        self.exports = ['bu']
        self.state = state
        if initial is None:
            initial = 2
        self.initial = initial

    def _cpost(self):
        """
        Full conditional posterior for Sigma_u as defined in equation 29 of
        Dong & Harris (2014)
        """
        pt = self.state.front
        for name in self.required:
            try:
                exec("{n} = state['{n}'}]".format(n=name))
            except KeyError:
                err = "Variable {} not found in trace".format(name)
                err += " at step {}".format(self.__class__)
                raise KeyError(err)
        Bus = np.dot(B, pt['Thetas'])
        bu = .5 * np.dot(Bus.T, Bus) + b0
        new_sigma_u = stats.invgamma.rvs(au, scale=bu)
        self.state['Sigma_u'] = new_sigma_u
        for name in self.exports:
            self.state[name] = eval(name)
        return state

class SAC_Upper(AbstractSampler):
    """
    Sampler for the full conditional of the spatially-aucorrelated 
    error parameter, called lambda.  

    This is the "upper-level" error spatial dependence coefficient in a Hierarchical
    Simultaneous Autoregressive model
    """
    def __init__(self, state=None, initial=.5):
        self.requires = ["M", "SAC_upper"]
        self.exports = [ 'parvals', 'density', 'S_sac_upper']
        self.state = state
        if initial is None:
            initial = .5
        self.initial = initial

    def _cpost(self, state=None):
        """
        Will be the full conditional posterior distribution for sac_upper as defined
        in equation 32 of Dong & Harris (2014), but is currently Unif(-1,1). 
        """
        if state is None:
            state= self.state
        cpt = self.state.front()
        for name in self.requires:
            try:
                exec("{n} = state['{n}']".format(n=name))
            except KeyError:
                err = "Variable {} not found in trace".format(name)
                err += " at step {}".format(self.__class__)
                raise KeyError(err)
        parvals = SAC_upper[:,0] #aka "lambda_exist"
        logdets = SAC_upper[:,1] #aka "log_detlambda"
        nsac_upper = len(parvals)
        iota = np.ones_like(parvals)
        
        uu = np.dot(pt['Thetas'].T, pt['Thetas'])
        uMu = np.dot(np.dot(pt['Thetas'].T, M), pt['Thetas'])
        Mu = np.dot(M, pt['Thetas'])
        uMMu = np.dot(Mu.T, Mu)

        S_sac_upper = uu*iota - 2*parvals*uMu + uMMu*parvals**2

        log_density = logdets - S_sac_upper/(2*pt['Sigma_u'])
        log_density = log_density - log_density.max()

        density = np.exp(log_density)
        
        new_sac_upper = isamp(density, grid=parvals) 
        state['SAC_upper'] = new_sac_upper
        for name in self.exports:
            state[name] = eval(name)
        return state

class SAC_Lower(AbstractSampler):
    """
    Sampler for the full SAC_Lower conditional posterior of an HSAR model

    This is the "lower-level" response spatial dependence coefficient in a
    Hierarchical Simultaneous Autoregressive model
    """
    def __init__(self, state=None, initial=.5):
        self.requires = ["e0", "ed", "e0e0", "eded", "e0ed", "Delta_u", "X", "SAC_Lowers"]
        self.exports = ['parvals', 'density', 'S_sac_lower']
        self.state = state
        if initial is None:
            initial = .5
        self.initial = initial

    def _cpost(self, state=None):
        """
        Will be the full conditional posterior distribution for SAC_Lower as defined
        in equation 31 of Dong & Harris (2014), but is currently Unif(-1,1). 
        """
        if state is None:
            state = self.state
        pt = self.state.front
        for name in self.requires:
            try:
                exec("{n} = state['{n}']".format(n=name))
            except KeyError:
                err = "Variable {} not found in trace".format(name)
                err += " at step {}".format(self.__class__)
                raise KeyError(err)
        parvals = SAC_Lowers[:,0] #values of sac_lower
        logdets = SAC_Lowers[:,1] #log determinant of laplacians
        nsac_lower = len(parvals)
        iota = np.ones_like(parvals)
        #have to compute eu-reltaed parameters
        beta_u, resids, rank, svs = la.lstsq(X, Delta_u)
        eu = Delta_u - np.dot(X, beta_u)
        eueu = np.dot(eu.T, eu)
        e0eu = np.dot(e0.T, eu)
        edeu = np.dot(ed.T, eu)

        S_sac_lower = (e0e0*iota + parvals**2 * eded + eueu 
                - 2*parvals*e0ed - 2*e0eu + 2*parvals*edeu)
        
        log_density = logdets - S_sac_lower/(2. * pt['Sigma_e'])
        adj = log_density.max()
        log_density = log_density - adj #downshift to zero?

        density = np.exp(log_density)

        new_sac_lower = isamp(density, grid=parvals)
        self.state['SAC_Lower'] = new_sac_lower
        for name in self.exports:
            self.state[name] = eval(name)
        return state

if __name__ == '__main__':
    samplers = []
    samplers.append(Betas())
    
