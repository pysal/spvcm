import numpy as np
import numpy.linalg as la
from ...utils import splogdet
from ...steps import metropolis

#############################
# SPATIAL SAMPLE METHODS    #
#############################

def logp_rho(state, val):
    """
    The logp for lower-level spatial parameters in this case has the same
    form as a multivariate normal distribution, sampled over the variance matrix, rather than over y.
    """
    st = state
    
    #must truncate in logp otherwise sampling gets unstable
    if (val < st.Rho_min) or (val > st.Rho_max):
        return np.array([-np.inf])
    
    PsiRho = st.Psi_1(val, st.W)
    PsiRhoi = la.inv(PsiRho)
    logdet = splogdet(PsiRho)
    
    eta = st.Y - st.XBetas - st.DeltaAlphas
    kernel = eta.T.dot(PsiRhoi).dot(eta) / st.Sigma2

    return -.5*logdet -.5 * kernel + st.Log_Rho0(val)

def logp_lambda(state, val):
    """
    The logp for upper level spatial parameters in this case has the same form
    as a multivariate normal distribution, sampled over the variance matrix,
    rather than over Y.
    """
    st = state

    #must truncate
    if (val < st.Lambda_min) or (val > st.Lambda_max):
        return np.array([-np.inf])

    PsiLambda = st.Psi_2(val, st.M)
    PsiLambdai = la.inv(PsiLambda)

    logdet = splogdet(PsiLambda)

    kernel = st.Alphas.T.dot(PsiLambdai).dot(st.Alphas) / st.Tau2

    return -.5*logdet - .5*kernel + st.Log_Lambda0(val)
