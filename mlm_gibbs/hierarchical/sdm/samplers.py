from pysal.spreg.sputils import spdot, spinv, splogdet
import scipy.linalg as la
import scipy.linalg as scla
from scipy import sparse as spar
from scipy.sparse import linalg as spla

# all params:
# beta : level 1 coefficients
# sigma: level 1 error variance
# alpha: level 2 intercept
# tau  : level 2 error variance (of alpha)
# gamma: level 2 coefficient
# rho  : lower SAR

# things to precompute:
# X.T @ X
# betacovprior*betameanprior
# Delta.T @ Delta

class Sigma(AbstractSampler):
    def __init__(self):
        pass
    def _cpost(self):
        st.e1 = st.y - spdot(st.Delta, st.alpha) - spdot(st.x,  st.beta)
        st.d1 = spdot(st.e1.T, st.e1) + sig2so * sig2vo #vo is initial nu,
                                                        # inital inverse chi-squared 
                                                        # dof parameter. 
        st.chi = np.random.chi(n+sig2vo, size=1)
        st.Sigma = np.sqrt(st.d1/st.chi)
        return st.Sigma

# precompute betacovprior @ betameanprior, call betacovmeanprior???

class Betas(AbstractSampler):
    def __init__(self):
        pass
    def _cpost_chol(self):
        covm = spinv(spdot(st.X.T, st.X)/st.Sigma**2 + betacovprior)
        mean = spdot(betacovprior, metameanprior)
        xyda = spdot(st.X.T, st.y - spdot(st.Delta,st.Alpha))
        move = betacov @ xyda / st.Sigma**2
        mean += move
        zs = np.random.normal(0, 1, size=st.p).reshape(st.p, 1)
        st.Betas = mean + spdot(scla.cholesky(covm).T,zs)
        return st.Betas

class Alphas(AbstractSampler):
    def __init__(self):
        pass

    def _cpost_chol(self):
        covm_kern = spdot(st.Delta.T, st.Delta) / st.Sigma**2
        covm_hetske = spdot(spdot(st.B.T, st.I_J), B) / st.Tau**2
        covm = spinv(alphakern + hetske)
        mean_kern = spdot(st.Delta.T, st.y - spdot(st.X, st.Betas)) / st.Sigma**2
        mean_hetske = spdot(st.B.T, spdot(st.Z, st.Gammas)) / st.Tau**2
        mean = spdot(covm, mean_kern + mean_hetske)
        zs = np.random.normal(0,1,size=st.J).reshape(st.J,1)
        st.Alphas = mean + spdot(scla.choleksy(covm).T, zs)
        return st.Alphas

class Tau(AbstractSampler):
    def __init__(self):
        pass

    def _cpost(self):
        e2 = spdot(st.B, st.Alphas) - spdot(st.Z, st.Gammas)
        d2 = spdot(e2.T, e2) + tau2so * tau2vo
        chi = np.random.chi(n+tau2vo, size=1)
        st.Tau = np.sqrt(d2/chi)
        return st.Tau

class Gammas(AbstractSampler):
    def __init__(self):
        pass

    def _cpost(self):
        covm = spinv(spdot(st.Z.T, st.Z)/st.Tau**2 + st.gammacovprior)
        mean_kern = spdot(spdot(st.Z.T, st.B), st.Alphas)/st.Tau**2 
        mean_kern += gammacovprior * gammameanprior
        mean = spdot(covm, mean_kern)
        zs = np.random.normal(0,1,size=st.q).reshape(st.q,1)
        st.Gammas = mean + spdot(scla.chol(covm).T, zs)

class Rho(AbstractSampler):
    def __init__(self):
        pass
    
    def _cpost(self):
        st.P = spar.csc_matrix(st.In - st.Rho * st.W)
        st.Rho, accepted = self._metropolis(self)
        if accepted:
            st.B = sp.csc_matrix(st.In - st.Rho * st.W)
        return st.Rho

    def _metropolis(self):
        denom = self._llike(st.P)
        new_val = self._propose(st.Rho)
        new_P = spar.csc_matrix(st.In - new_val * st.W)
        num = self._llike(new_P)
        
        uval = np.random.random()
        diff = num - denom
        if diff > np.exp(1):
            pp = 1;
        else:
            pp = np.min(1, np.exp(diff))
        
        if uval < pp:
           returnval = new_val
           self.acc += + 1
           accepted = True
        else:
            returnval = st.Rho
            accepted = False
        
        ### Since we've 'finished' here, we need stepcount + 1
        ### stepcount wont increment until this draw completes
        accept_rate = self.acc / (self.state.steps + 1)
        
        if ar < .4:
            self.step /= 1.1
        elif ar > .6:
            self.step *= 1.1
        return returnval, accepted

    def _llike(self, val):
        ret = splogdet(val)
        pazg = spdot(val, st.Alphas) - spdot(st.Z, st.Gammas)
        ret += .5/st.Sigma**2 * spdot(pazg.T, pazg) + st.rhoprior
        return ret 
    
    def _propose(self, val):
        while True:
            proposed = val + self.step * np.random.normal() 
            if -1 < proposed < 1:
                break
        return proposed
