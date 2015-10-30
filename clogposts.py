import numpy as np
from numpy import linalg as la
from scipy import stats
import copy

def beta(trace):
    """
    Uses:
    --------------
    Stochastics:
           sigma_e
           rho
           u
    Statics:
           XtX
           In
           W
           invT0
           y
           Z
           X
           T0M0
           p
    Intermediates:

    Yields:
    ---------
    new_betas
    Ay
    A
    """
    exports = ['Ay', 'A']
    s = trace.Statics
    pt = trace.current_point()
    VV = s['XtX'] / pt['sigma_e'] + s['invT0']
    v_betas = la.inv(VV) #conditional posterior variance matrix
    A = In - pt['rho'] * s['W']
    Ay = np.dot(s['A'], s['y'])
    Delta_u = np.dot(s['Delta'], pt['u']) #recall, HSAR.R labels Delta from paper as Z
    lprod = np.dot(X.T, (Ay - Delta_u)/pt['sigma_e'] + s['T0M0']) 
    m_betas = np.dot(v_betas, lprod) #conditional posterior mean
    new_betas = np.random.multivariate_normal(m_betas.flatten(), v_betas, p)
    trace.Memo['A'] = A
    trace.Memo['Ay'] = Ay
    return trace.update('betas', new_betas) #outputs new betas

def u(trace):
    """
    Uses:
    ------------
    Stochastics:
        lam
        sigma_e
        sigma_u
        betas
        u
    Statics:
        Ij
        M
        Z
        X
        y
        J
    Intermediates:
        Ay

    Yields:
    =========
        new_u
        Xb
        B

    """
    pt = trace.current_point()
    B = Ij - pt['lam'] * M
    v_u = np.dot(Z.T, Z)/pt['sigma_e'] + np.dot(B.T, B)/pt['sigma_u']
    v_ui = la.inv(v_u) #conditional posterior variance matrix
    Xb = np.dot(X, pt['betas'])
    lprod = np.dot(Z.T, Ay - Xb) / pt['sigma_e']
    m_u = np.dot(v_u, lprod)
    new_u = np.random.multivariate_normal(m_u, v_u, J)
    return trace.update('u', new_u)

def sigma_e(trace):
    pt = trace.current_point()
    Zu = np.dot(Delta, pt['u'])
    e = Ay - Zu - Xb
    de = .05 * np.dot(e, e.T)
    new_sigma_e = stats.invgamma.rvs(ce, scale=de)
    return trace.update('sigma_e', new_sigma_e)

def sigma_u(trace):
    pt = trace.current_point()
    Bus = np.dot(B, pt['u'])
    bu = np.dot(Bus, Bus.T)/2. + b0
    new_sigma_u = stats.invgamma.rvs(au,scale=bu)
    return trace.update('sigma_u', new_sigma_u)

#def rho_sample(rhos, I=In, W=W, e0e0=e0e0, eded=eded, e0ed=e0ed, 
#                                 eueu=None, e0eu=None, edeu=None, sig=None):
#    density = logcpost_rho(rhos, I, W, e0e0, eded, e0ed, eueu, e0eu, edeu, sig)
#    integral, accuracy
#    normalized = density / float(integral)
#
#def logcpost_rho(rhos, I=In, W=W, e0e0=e0e0, eded=eded, e0ed=e0ed, 
#                                 eueu=None, e0eu=None, edeu=None, sig=None):
#    nrhos = rhos.shape[0]
#    rvals = rhos[:,0].reshape((nrhos, 1))
#    rdets = rhos[:,1].reshape(rvals.shape)
#
#    iota = np.ones_like(rvals)
#
#    S_rho = e0e0*iota + rvals**2*eded + eueu - 2*rvals*e0ed - 2*e0eu + e*rvals*edeu
#
#    log_den = log_detrho - S_rho/(2. * sig)
#    log_den = log_den - log_den.max()
#
#    return  np.exp(log_den)
