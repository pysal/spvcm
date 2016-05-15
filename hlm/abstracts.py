from __future__ import division
from .utils import Namespace as NS
from scipy import stats
import numpy as np

class Gibbs(object):
    """
    A Gibbs Sampler manager.
    """
    def __init__(self, *samplers, **kwargs):
        self.samplers = []
        self.trace = NS()
        self._state = kwargs.pop('state', NS())
        self.add_steps(*samplers)
        self.steps = 0
        self._verbose = kwargs.pop('verbose', False)
    
    def __getitem__(self, val):
        return self._state[val]

    @property
    def var_names(self):
        return [s.__name__ for s in self.samplers]

    @property
    def cycles(self):
        """
        How many cycles the sampler has fully completed
        """
        return self.steps // len(self.samplers)
    
    def sample(self, cycles):
        """
        Move an arbitrary number of steps or cycles forward in the sampler. 

        Parameters
        ----------
        cycles  :   int
                    the number of cycles, i.e. full rotations of all conditional
                    posteriors, to take 
        """
        while cycles > 0:
            if self._verbose:
                print('starting sampling cycle {}'.format(self.cycles))
            for sampler in self.samplers:
                if self._verbose > 1:
                    print('\tsampling from {}'.format(sampler.__name__))
                sampler() #call is sample
                self.trace[sampler.__name__].append(self._state[sampler.__name__]) 
                self.steps += 1
            cycles -= 1

    def front(self, *names):
        """
        grab the most recent values of parameters from either this cycle or the
        previous cycle
        """
        if names is ():
            names = self.var_names
        return NS(**{name:self._state[name] for name in names})
    
    def add_steps(self, *steps):
        """
        Add any number of sampling steps to the Gibbs object

        Arguments
        ---------
        steps    :   any number of samplers to add to the Gibbs sampler
        """
        for step in steps:
            step.sampler = self
            self.samplers.append(step)
            self._state.update({step.__name__:step.initial})
            self.trace.update({step.__name__:[step.initial]})

class Abstract_Step(object):
    """
    One sampling step. This class is not intended to be used, but should be
    inherited. 

    The main idea is that classes that inherit from this will define some
    _cpost function that draws from a conditional posterior. 
    """
    def __init__(self, sampler=None,name=None, **params):
        self.sampler = sampler
        if name is None:
            name = self.__class__.__name__
        self.__name__ = name
    
    def __next__(self, *args, **kwargs):
        """
        Return the next draw from the conditional posterior using some sampling
        technique.
        """
        return self._draw()
    
    def __call__(self, *args, **kwargs):
        """
        equivalent to calling __next___
        """
        return self.__next__(*args, **kwargs)

    def _draw(self):
        """
        Take generate one draw from the conditional posterior distribution,
        conditioned on current sampler state
        """
        raise NotImplementedError

    def _logp(self, val):
        """
        Compute the log of the conditional pdf at a value
        """
        return NotImplementedError

class Metropolis_Mixin(object):
    """
    A descriptive class describing attributes needed for metropolis sampling
    steps.

    Parameters
    ----------
    default_move    :   float
                        the default scale parameter for the proposal
                        distribution
    adapt_rate      :   float
                        the scaling factor by which to adapt the metropolis
                        rate. This should be greater than one if the metropolis
                        is stablized towards the center.
    lower_bound     :   float
                        the target lower bound for the acceptance rate
    upper_bound     :   float
                        the target upper bound for the acceptance rate
    proposal        :   scipy.stats.rv_continuous
                        a distribution that supports both the computation of
                        random values using proposal.rvs and the computation of
                        the log pdf using proposal.logpdf.
    """
    def __init__(self, default_move=1, adapt_rate=1, 
                 lower_bound=.4, upper_bound=.6, proposal=stats.norm):
        self.n_accepted = 0
        self._current_logp = None
        self._move_size= default_move
        self._adapt_rate = adapt_rate
        self._lower = lower_bound
        self._upper = upper_bound
        self._proposal = proposal
        self._symproposal=True

    def _metropolis(self, old):
        """
        sample using metropolis-hastings. This is done by accepting a move from some
        current parameter value to some new parameter value with the probability:

        A = P(new) / P(current) * f(current | new) / f(new | current)

        where the first term is the ratio of the pdfs new and current, and f is the
        distribution of the proposal. In logs, this is:

        log(A) = log(P(new)) - log(P(current)) + (log(f(current | new)) - log(f(new | current)))
        """
        if self._current_logp is not None:
            ll_old = self._current_logp
        else:
            ll_old = self._logp(old)
        new, forward_logp, backward_logp = self._propose(old)
        ll_new = self._logp(new) 

        diff = ll_new - ll_old
        diff += (backward_logp - forward_logp)
        
        ratio = np.exp(diff)

        uval = np.random.random()
       
        pp = np.min((1, ratio))
        
        if uval < pp:
           returnval = new
           self.n_accepted += 1
        else:
            returnval = old
        self._adapt()

        return returnval

    def _propose(self, current_value):
        """
        This proposes a move and provides the proposal ratio of making the move.
        Formally, the proposal ratio for some proposal distribution g() is:
        
        g(current | new) / g(new | current)
        
        where current is the proposed new parameter value and new is the current
        parameter value. 

        Arguments
        ---------
        value   :   float
                    current value of the parameter
        move_size       :   float
                            Overrides the current variance of the proposal. If not
                            passed, self._move_size is used.
        distribution    :   scipy.stats.rv_continuous
                            a scipy statistical distribution that supports the
                            drawing of random variables using distribution.rvs
                            and the computation of the log density via
                            distribution.logpdf
        symmetric       :   bool
                            if True, the proposal probabilities are not
                            computed and the proposal ratio is fixed to one.
        """
        new = self._proposal.rvs(loc=current_value, scale=self._move_size)
        if self._symproposal:
            forward = backward = 1
        else:
            forward = self._proposal.logpdf(new, loc=current_value, 
                                            scale=self._move_size)
            backward = self._proposal.logpdf(value, loc=new, 
                                             scale=self._move_size)
        return new, forward, backward

    def _adapt(self):
        """
        This adapts the step size of the metropolis sampler according to the
        acceptance rate. If the acceptance rate is too high, larger steps are
        taken to lower acceptance rate. 
        If the acceptance rate is too low, smaller steps are taken to increase
        the acceptance rate. 

        This is essentially a no-op if self._adapt_rate is 1.
        """
        self.accept_rate = self.n_accepted / (self.sampler.cycles + 1)
        
        if self.accept_rate < self._lower:
            self._move_size *= self._adapt_rate 
        elif self.accept_rate > self._upper:
            self._move_size /= self._adapt_rate
