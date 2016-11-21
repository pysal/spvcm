from __future__ import division
import warnings
import numpy as np
import copy
import scipy.linalg as scla
import scipy.sparse as spar
from scipy import stats
from scipy.spatial import distance as d
from .utils import explode_stack, nexp
from .sample import logp_phi_j
from ...abstracts import Sampler_Mixin, Trace, Hashmap
from ...steps import Metropolis, Slice
from ...utils import chol_mvn

#thanks, python 2
from ...verify import center as verify_center, covariates as verify_covariates

class MSVC(Sampler_Mixin):
    """
    A class to initialize a Multi-spatially-varying Coefficient model
    """
    def __init__(self,
                 #data parameters
                 Y, X=None, Z=None, coordinates=None, n_samples=1000, n_jobs=1,
                 # prior configuration parameters
                 extra_traced_params = None,
                 priors=None,
                 starting_values = None,
                 configs=None,
                 correlation_function=nexp,
                 dmetric='euclidean',
                 verbose=False,
                 center=True,
                 constant='local',
                 rescale_dists=True):
        super(MSVC, self).__init__()
        if X is None and Z is None:
            raise UserWarning('At least one of X or Z must be provided.')

        if coordinates is None:
            raise UserWarning('Missing `coordinates` array, cannot fit.')

        self.state = Hashmap()
        st = self.state
        self.traced_params = (['Gammas', 'Zetas', 'Mus',
                                'Sigma2s', 'Phis', 'Tau2'])
        if Z is not None:
            if center:
                Z = Z - Z.mean()
            if constant.lower().startswith('gl') or constant.lower().startswith('bo'):
                Z = verify_covariates(Z)
            st.Z = Z
            st.pz = Z.shape[-1]
            self.state._has_Z = True
        else:
            self.traced_params = (['Zetas', 'Mus', 'Sigma2s', 'Phis', 'Tau2'])
            self.state.ZGammas = 0
            self.state._has_Z = False
        if extra_traced_params is not None:
            if isinstance(extra_traced_params, str):
                extra_traced_params = [extra_traced_params]
            self.traced_params.extend(extra_traced_params)
        self.trace = Trace(**{param:[] for param in self.traced_params})
        st = self.state
        self.state.correlation_function = correlation_function
        self.verbose = verbose


        st.Y = Y
        if X is None and (constant.lower().startswith('lo')
                          or constant.lower().startswith('bo')):
            X = np.ones_like(Y)
        else:
            if center:
                X = verify_center(X)
            if constant.lower().startswith('lo') or constant.lower().startswith('bo'):
                X = verify_covariates(X)
        st.X = X
        n,p = st.X.shape
        st.Xc = explode_stack(st.X)
        st.n = n
        st.p = p

        st.coordinates = coordinates - coordinates.mean(axis=0)
        st._dmetric = dmetric
        if isinstance(st._dmetric, str):
            st.pwds = d.squareform(d.pdist(st.coordinates, metric=st._dmetric))
        elif callable(st._dmetric):
            st.pwds = st._dmetric(st.coordinates)

        st.max_dist = st.pwds.max()
        if rescale_dists:
            st.pwds = st.pwds/st.max_dist
            st._old_max = st.max_dist
            st.max_dist = 1.

        if priors is None:
            priors = dict()
        if starting_values is None:
            starting_values = dict()
        if configs is None:
            configs = dict()

        self._setup_priors(**priors)
        self._setup_starting_values(**starting_values)
        self._setup_configs(**configs)
        self._verbose = verbose
        self.cycles = 0

        if n_samples > 0:
            try:
                self.sample(n_samples, n_jobs=n_jobs)
            except (np.linalg.LinAlgError, ValueError) as e:
                warnings.warn('Encountered the following LinAlgError. '
                     'Model will return for debugging. \n {}'.format(e))

    def _setup_priors(self, Tau2_a0 = .001, Tau2_b0 = .001,
                            Mus_mean0 = None, Mus_cov0 = None,
                            Gammas_mean0 = None, Gammas_cov0 = None,
                            Phi_shape0_list = None, Phi_rate0_list = None,
                            Sigma2_a0_list = None, Sigma2_b0_list = None):
        st = self.state
        st.Tau2_a0 = Tau2_a0
        st.Tau2_b0 = Tau2_b0

        if Mus_mean0 is None:
            Mus_mean0 = np.zeros((st.p, 1))
        if Mus_cov0 is None:
            Mus_cov0 = np.eye(st.p)

        st.Mus_cov0 = Mus_cov0
        st.Mus_mean0 = Mus_mean0

        if st._has_Z:
            if Gammas_mean0 is None:
                Gammas_mean0 = np.zeros((st.pz, 1))
            if Gammas_cov0 is None:
                Gammas_cov0 = np.eye(st.pz)
        else:
            Gammas_cov0 = 0
            Gammas_mean0 = 0
        st.Gammas_cov0 = Gammas_cov0
        st.Gammas_mean0 = Gammas_mean0

        if Phi_shape0_list is None:
            Phi_shape0_list = [1]*st.p
        elif isinstance(Phi_shape0_list, (float, int)):
            Phi_shape0_list = [Phi_shape0_list] * st.p
        st.Phi_shape0_list = Phi_shape0_list

        if Phi_rate0_list is None:
           Phi_rate0_list = ([Phi_shape0 /
                                 ((-.5*st.pwds.max() / np.log(.05)))
                                 for Phi_shape0 in Phi_shape0_list])
        elif isinstance(Phi_rate0_list, (float,int)):
            Phi_rate0_list = [Phi_rate0_list]*st.p
        st.Phi_rate0_list = Phi_rate0_list

        if Sigma2_a0_list is None:
            Sigma2_a0_list = [2] * st.p
        elif isinstance(Sigma2_a0_list, (float,int)):
            Sigma2_a0_list = [Sigma2_a0_list]*st.p
        st.Sigma2_a0_list = Sigma2_a0_list

        if Sigma2_b0_list is None:
            Sigma2_b0_list = [2]*st.p
        elif isinstance(Sigma2_b0_list, (float,int)):
            Sigma2_b0_list = [Sigma2_b0_list]*st.p
        st.Sigma2_b0_list = Sigma2_b0_list

    def _setup_configs(self, Phi_method = 'met', Phi_configs=None, **uncaught):
        """
        Ensure that the sampling configuration is set accordingly.
        """
        if Phi_method.lower().startswith('met'):
            method = Metropolis
        elif Phi_method.lower().startswith('slice'):
            method = Slice
        else:
            raise Exception('`Phi_method` option not understood. `{}` provided'.format(Phi_method))
        if uncaught != dict() and Phi_configs is None:
            Phi_configs = [uncaught] * self.state.p
        elif Phi_configs is not None and uncaught == dict():
            if isinstance(Phi_configs, dict):
                Phi_configs = [Phi_configs] * self.state.p
            elif isinstance(Phi_configs, list):
                conformal = len(Phi_configs) == self.state.p
                if not conformal:
                    raise Exception('Length of passed `Phi_configs` does not'
                                    ' match the number of local variables!'
                                    ' Refusing to guess.')
            else:
                raise TypeError('Type of `Phi_configs` not understood. Must be'
                                'dict containing configurations for all '
                                'processes or list of dicts containing configs '
                                'for each process. Recieved object of type:\n{}'
                                .format(type(Phi_configs)))
        elif Phi_configs is None and uncaught == dict():
            Phi_configs = [dict()] *self.state.p
        else:
            raise Exception('Uncaught options {} passed in addition to '
                            '`Phi_configs` {}.'.format(uncaught, Phi_configs))

        self.configs = Hashmap()
        self.configs.Phis = [method('Phi_{}'.format(j), logp_phi_j, **confs)
                                for j,confs in enumerate(Phi_configs)]

    def _setup_starting_values(self, Phi_list = None, Sigma2_list=None,
                                     Mus = None, Gammas=None,
                                     Zetas = None, Tau2 = 2):
        """
        set up initial values for the sampler
        """
        st = self.state

        if Phi_list is None:
            Phi_list = [3*Phi_shape0_i / phi_rate0_i
                            for Phi_shape0_i, phi_rate0_i
                            in zip(st.Phi_shape0_list, st.Phi_rate0_list)]
        elif isinstance(Phi_list, (float, int)):
            Phi_list = [Phi_list] * st.p
        self.state.Phi_list = Phi_list
        if Sigma2_list is None:
            Sigma2_list = [1]*st.p
        elif isinstance(Sigma2_list, (float,int)):
            Sigma2_list = [Sigma2_list]*st.p
        self.state.Sigma2_list = Sigma2_list

        if Mus is None:
            Mus = np.zeros((1,st.p))
        self.state.Mus = Mus

        if Gammas is None and self.state._has_Z:
            Gammas = np.zeros((st.pz, 1))
        elif not self.state._has_Z:
            Gammas = 0
        self.state.Gammas = Gammas

        if Zetas is None:
            Zetas = np.zeros((st.n * st.p, 1))
        self.state.Zetas = Zetas

        self.state.Tau2 = Tau2

    def _fuzz_starting_values(self):
        st = self.state
        if st._has_Z:
            st.Gammas += np.random.normal(0,5, size=st.Gammas.shape)
        st.Mus += np.random.normal(0,5, size=st.Mus.shape)
        st.Phi_list = [phi + np.random.uniform(0,10) for phi in st.Phi_list]
        st.Sigma2_list = [sigma2 + np.random.uniform(0,10) for sigma2 in st.Sigma2_list]
        st.Zetas += np.random.normal(0,5, size=st.Zetas.shape)
        st.Tau2 += np.random.uniform(0,10)


    def _finalize(self):
        st = self.state
        st.In = np.identity(st.n)
        st.XtX = st.X.T.dot(st.X)
        st.XctXc = st.Xc.T.dot(st.Xc)
        if st._has_Z:
            st.ZtZ = st.Z.T.dot(st.Z)
            st.ZGammas = st.Z.dot(st.Gammas)

        st.H_list = [st.correlation_function(phi, st.pwds)
                     for phi in st.Phi_list]
        st.Hi_list = [np.linalg.inv(Hj) for Hj in st.H_list]
        st.Zeta_list = np.array_split(st.Zetas, st.p)
        st.Sigma2_an_list = [self.state.n/2.0 + a0
                          for a0 in st.Sigma2_a0_list]
        st.Tau2_an = self.state.n / 2.0 + st.Tau2_a0
        st.Mus_cov0i = np.linalg.inv(st.Mus_cov0)
        if st._has_Z:
            st.Gammas_cov0i = np.linalg.inv(st.Gammas_cov0)
        st.XcZetas = st.Xc.dot(st.Zetas)
        st.H = scla.block_diag(*st.H_list)
        st.Hi = scla.block_diag(*st.Hi_list)

    def _iteration(self):
        st = self.state
        ##The sample step for mu_beta in a multi-process SVC
        #N(Sn.dot(Mn), Sn),
        #where Mn is Xt(Y - ZGammas - XcZetas)/Tau2 + S0^-1m0
        #and Sn is (XtX/Tau2 + S0^-1)^-1

        Sni = st.XtX / st.Tau2 + st.Mus_cov0i
        Sn = np.linalg.inv(Sni)
        Mn = (st.X.T.dot(st.Y - st.ZGammas -  st.XcZetas)) / st.Tau2
        Mn += (st.Mus_cov0i.dot(st.Mus_mean0))
        st.Mus = chol_mvn(Sn.dot(Mn), Sn)
        st.XMus = st.X.dot(st.Mus)

        #The sample step for tau in a multi-process SVC
        #IG(an,bn)
        #where an is n/2+a0
        #and bn is (Y-XMus-XcZetas - ZGammas)**2/2 + b0,
        # where **2 is the matrix #square
        st.eta = st.Y - st.XMus - st.XcZetas - st.ZGammas
        st.Tau2_bn = st.eta.T.dot(st.eta) / 2 + st.Tau2_b0
        st.Tau2 = stats.invgamma.rvs(st.Tau2_an, scale=st.Tau2_bn)

        #The sample step for the full vector of local random effects, Zeta.
        #N(Sigma_zn.dot(zn), Sigma_zn)
        #where zn is Xc'(Y - ZGamma - XMu) / Tau2
        # and Sigma_zn is Xc'Xc/Tau2 + Hi
        zn = st.Xc.T.dot(st.Y - st.ZGammas - st.XMus) / st.Tau2
        Sigma_zni = st.XctXc / st.Tau2 + st.Hi
        Sigma_zn = np.linalg.inv(Sigma_zni)
        st.Zetas = chol_mvn(Sigma_zn.dot(zn), Sigma_zn)
        st.XcZetas = st.Xc.dot(st.Zetas)
        st.Zeta_list = np.split(st.Zetas, st.p)

        # The sample step for the global effects, gamma
        # N(Sigma_gn.dot(gn), Sigma_gn)
        #  where gn is Z'(Y - XMus - XcZetas)/Tau2 + S0g0
        #  and Sigma_gn is Z'Z/Tau2 + S0
        if st._has_Z:
            eta_MuZeta = st.Y - st.XMus - st.XcZetas
            gn = st.Z.T.dot(eta_MuZeta) / st.Tau2
            st.gn = gn + st.Gammas_cov0i.dot(st.Gammas_mean0)

            Sgni = st.ZtZ / st.Tau2 + st.Gammas_cov0i
            st.Sgn = np.linalg.inv(Sgni)
            st.Gammas = chol_mvn(st.Sgn.dot(st.gn), st.Sgn)
            st.ZGammas = st.Z.dot(st.Gammas)

        for j, step in enumerate(self.configs.Phis):
            st.j = j
            st.update({'Phi_{}'.format(j):st.Phi_list[j]})
            ### Drawing Phi frmo its non-analytic posterior
            st.Phi_list[j] = step(self.state)
            st.H_list[j] = st.correlation_function(st.Phi_list[j], st.pwds)
            st.Hi_list[j] = np.linalg.inv(st.H_list[j])

            ### Drawing sigma2 from its IG(a,b) posterior

            Hj = st.H_list[j]
            Hji = st.Hi_list[j] #could we get this out of Zeta's Hi?
            Zeta_j = st.Zeta_list[j]
            Phi_j = st.Phi_list[j]
            a0_j = st.Sigma2_a0_list[j]
            b0_j = st.Sigma2_b0_list[j]
            an_j = st.Sigma2_an_list[j]

            bn_j = Zeta_j.T.dot(Hji).dot(Zeta_j) / 2 + b0_j
            st.Sigma2_list[j] = stats.invgamma.rvs(an_j, scale=bn_j)

        st.H = scla.block_diag(*[Hj * Sigma2j for Hj, Sigma2j in
                                 zip(st.H_list, st.Sigma2_list)])
        st.Hi = scla.block_diag(*[Hji/Sigma2j for Hji, Sigma2j in
                                  zip(st.Hi_list, st.Sigma2_list)])
        st.Sigma2s = np.squeeze(st.Sigma2_list)
        st.Phis = np.squeeze(st.Phi_list)
