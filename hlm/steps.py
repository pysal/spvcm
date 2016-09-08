import numpy as np
MAX_SLICE = 1000

def inversion(pdvec, grid):
    """
    sample from a probability distribution vector, according to a grid of values
    
    Parameters
    -----------
    pdvec   :   np.ndarray
                a vector of point masses that must sum to one. in theory, an
                approximation of a continuous pdf
    grid    :   np.ndarray
                a vector of values over which pdvec is evaluated. This is the
                bank of discrete values against which the new value is drawn
    """
    if not np.allclose(pdvec.sum(), 1):
        pdvec = pdvec / pdvec.sum()
    cdvec = np.cumsum(pdvec)
    a = 0
    while True:
        a += 1
        rval = np.random.random()
        topidx = np.sum(cdvec <= rval) -1
        if topidx >= 0:
            return grid[topidx]

def metropolis(state, current, proposal, logp, configs):
    """
    Sample using metropolis hastings in its simplest form

    Parameters
    ----------
    state   :   Namespace
                state required to evaluate the logp of the parameter
    current :   float/int
                current value of the parameter
    proposal:   scipy random distribution
                distribution that can has both `logpdf` and `rvs` methods
    logp    :   callable(state, value)
                function that can compute the current log of the probability
                distribution of the value provided conditional on the state
    configs :   Namespace
                must contain `jump` attribute, which is the scale parameter for
                the proposal.

    Returns
    --------
    new (or current) parameter value, and boolean indicating whether or not a
    new proposal was accepted.
    """
    current_logp = logp(state, current)
    new_val = proposal.rvs(loc=current, scale=configs.jump)
    new_logp = logp(state, new_val)
    forwards = proposal.logpdf(new_val, loc=current, scale=configs.jump)
    backward = proposal.logpdf(current, loc=new_val, scale=configs.jump)
    
    hastings_factor = backward - forwards
    r = new_logp - current_logp + hastings_factor
    
    r = np.min((1, np.exp(r)))
    u = np.random.random()
    
    if u < r:
        outval = new_val
        accepted = True
    else:
        outval = current
        accepted = False
    return outval, accepted

def slicer(state, current, logp, configs):
    """
    Implements slice sampling on a bounded log-concave parameter. This allows
    for no stepping-out to be needed, since the bounds provide the X range and
    the function provides the Y range. 
    """
    current_logp = logp(state, current)
    low, hi = configs.bounds
    n_iterations = 0
    # p. 712 of Neal defines this auxiliary variable on the log scale
    slice_height = current_logp - np.random.exponential()
    while True:
        candidate = np.random.uniform(low, hi)
        cand_logp = logp(state, candidate)
        if slice_height <= cand_logp:
            return candidate, True
        n_iterations += 1
        if n_iterations > MAX_SLICE:
            warn('Slicing is failing to find an effective candidate. '
                 'Using a metropolis update.', stacklevel=2)
            return metropolis(state, current, logp, configs)
