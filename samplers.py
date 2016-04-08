from __future__ import division

class Gibbs(object):
    """
    A Gibbs Sampler manager.
    """
    def __init__(self, *samplers, **kwargs):
        self.samplers = list(samplers)
        self._state = kwargs.pop('state', globals())
        self.trace = {k:[self._state[k]] for k in self.var_names}
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
        self.sample(steps=1)

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
        for i in range(len(samplers) - step):
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
            for smp in self.samplers:
                smp()
                self.trace[smp.__name__].append(self._state[smp.__name__])
                self.steps += 1

    @property
    def front(self, *names):
        """
        grab the most recent values of parameters from either this cycle or the
        previous cycle
        """
        if names is ():
            names = self.var_names
        return {name:self._state[name] for name in names}

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
        return {name:vals[name] for name in names}

    @property
    def previous(self, *names):
        """
        grab sampled values for the previous fully-completed cycle of the
        sampler. 
        """
        if names is ():
            names = self.var_names
        return {name: self.trace[name] for name in names}

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
