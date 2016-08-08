from warnings import warn as Warn
from datetime import datetime as dt

class Sampler_Mixin(object):
    def __init__(self):
        pass

    def sample(self, n_samples):
        """
        Sample from the joint posterior distribution defined by all of the
        parameters in the gibbs sampler. 

        Parameters
        ----------
        n_samples      :   int
                        number of samples from the joint posterior density to
                        take
        pop         :   bool
                        whether to eject the trace from the sampler. If true,
                        this function will return a namespace containing the
                        results of the sampler during the run, and the sampler's
                        trace will be refreshed at the end. 

        Returns
        -------
        updates all values in place, may return trace of sampling run if pop is
        True
        """
        self._finalize_invariants()
        _start = dt.now()
        try:
            while n_samples > 0:
                if (self._verbose > 1) and (n_samples % 100 == 0):
                    print('{} Draws to go'.format(n_samples))
                self.draw()
                n_samples -= 1
        except KeyboardInterrupt:
            Warn('Sampling interrupted, drew {} samples'.format(self.cycles))
        finally:
            _stop = dt.now()
            if not hasattr(self, 'total_sample_time'):
                self.total_sample_time = _stop - _start
            else:
                self.total_sample_time += _stop - _start 

    def draw(self):
        """
        Take exactly one sample from the joint posterior distribution
        """
        self._sample()
        for param in self.traced_params:
            getattr(self.trace, param).append(getattr(self.state, param))
        self.cycles += 1
