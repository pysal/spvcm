from __future__ import division
from .utils import Namespace as NS
from scipy import stats

class Gibbs(object):
    """
    A Gibbs Sampler manager.
    """
    def __init__(self, *samplers, **kwargs):
        self.samplers = list(samplers)
        self._state = kwargs.pop('state', globals())
        self.trace = NS(**{k:[self._state[k]] for k in self.var_names})
        self.steps = 0
    
    def __getitem__(self, val):
        return self._state[val]

    @property
    def var_names(self):
        return [type(s).__name__ for s in self.samplers]

    @property
    def position(self):
        """
        The current position (step) of the sampler within a cycle.
        """
        return self.steps % len(self.samplers)
    
    def step(self):
        """
        Take one step within a cycle of the sampler
        """
        self.sample(steps=1, cycles=0)

    @property
    def cycles(self):
        """
        How many cycles the sampler has fully completed
        """
        return self.steps // len(self.samplers)
    
    def cycle(self, finish=False):
        """
        Either finish the current full cycle of the sampler (i.e. return to position 0)
        or complete one full cycle of the sampler (i.e. return to current
        position)

        Arguments
        ---------
        finish  :   bool
                    if True, will step until the position is zero.
                    if False, will step until one full cycle has completed from
                    the current position. 
        """
        if finish:
            to_take = len(self.samplers - self.position)
        else:
            to_take = len(self.samplers)
        for _ in range(to_take):
            self.step()
        if finish:
            assert self.position == 0

    def sample(self, steps=0, cycles=1):
        """
        Move an arbitrary number of steps or cycles forward in the sampler. 

        Parameters
        ----------
        steps   :   int
                    the number of steps, i.e. sequential conditional posterior
                    draws, to take
        cycles  :   int
                    the number of cycles, i.e. full rotations of all conditional
                    posteriors, to take 
        """
        to_take = steps + cycles * len(self.samplers)
        while to_take > 0:
            if self._verbose:
                if self.position is 0:
                    print('starting sampling cycle {}'.format(self.cycles))
                if self._verbose > 1:
                    current_name = self.samplers[self.position].__name__
                    print('\tsampling from {}'.format(current_name))
            smp = self.samplers[self.position] #sampler in current position
            smp() #call is sample
            self.trace[smp.__name__].append(self._state[smp.__name__]) 
            self.steps += 1
            to_take -= 1

    @property
    def front(self, *names):
        """
        grab the most recent values of parameters from either this cycle or the
        previous cycle
        """
        if names is ():
            names = self.var_names
        return NS(**{name:self._state[name] for name in names})

    @property
    def current(self, *names):
        """
        grab sampled values for the current cycle of the sampler
        """
        vals = {k:None for k in self.var_names}
        current = self.front
        for name in self.var_names:
            vals[name] = current[name]
        if names is ():
            names = self.var_names
        return NS(**{name:vals[name] for name in names})

    @property
    def previous(self, *names):
        """
        grab sampled values for the previous fully-completed cycle of the
        sampler. 
        """
        if names is ():
            names = self.var_names
        return NS(**{name: self.trace[name][self.cycles-1] for name in names})

class AbstractSampler(object):
    """
    One sampling step. This class is not intended to be used, but should be
    inherited. 

    The main idea is that classes that inherit from this will define some
    _cpost function that draws from a conditional posterior. 
    """
    def __init__(self, state=None,name=None, **params):
        if state == None:
            state = globals()
        self.state = state
        if name is None:
            name = self.__class__.__name__
        self.__name__ = name
    
    def __next__(self, *args, **kwargs):
        """
        get another draw from the sampling function
        """
        return self.__call__(*args, **kwargs) 
    
    def __call__(self, *args, **kwargs):
        """
        equivalent to calling __next__
        """
        return self._cpost(*args, **kwargs)

    def _cpost(self):
        """
        The conditional posterior to sample from. Makes no sense for the
        abstract class
        """
        return 1

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
                 lower_bound=.4, upper_bound=.6, proposal=stats.normal):
        self.accepted = 0
        self._current_logp = self._logp(self.initial)
        self._move_size= default_move
        self._adapt_rate = adapt_rate
        self._lower = lower_bound
        self._upper = upper_bound
        self._proposal = proposal

    def _metropolis(self, value):
        """
        sample using metropolis-hastings. This is done by accepting a move from some
        current parameter value to some new parameter value with the probability:

        A = P(new) / P(current) * f(current | new) / f(new | current)

        where the first term is the ratio of the pdfs new and current, and f is the
        distribution of the proposal. In logs, this is:

        log(A) = log(P(new)) - log(P(current)) + (log(f(current | new)) - log(f(new | current)))
        """
        ll_now = self._current_logp
        new, forward_logp, backward_logp = self._propose(value)
        ll_new = self._logp(new) 

        diff = ll_now - ll_new
        diff += (forward_logp - backward_logp)
        
        A = np.exp(diff)

        uval = np.random.random()
       
        pp = np.min(1, A)
        
        if uval < pp:
           returnval = new_val
            self.accepted += 1
        else:
            returnval = value
            self.accepted += 1
        self._adapt()

        return returnval

    def _logp(self, value):
        """
        This should return the log of the pdf being sampled. 
        """
        return 1

    def _propose(self, value, move_size=None, distribution=None, symmetric=False):
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
        if move_size is None:
            move_size = self._move_size
        if distribution is None:
            distribution = self._proposal
        new = distribution.rvs(loc=value, scale=move_size)
        if symmetric:
            forward = backward = 1
        else:
            forward = distribution.logpdf(new, loc=value, scale=move_size)
            backward = distribution.logpdf(value, loc=new, scale=move_size)
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
        self.accept_rate = self.n_accepted / (self.state.cycles + 1)
        
        if self.accept_rate < self._lower:
            self._move_size /= self._adapt_rate 
        elif self.accept_rate > self._upper:
            self._move_size *= self._adapt_rate
