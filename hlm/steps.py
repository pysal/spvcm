def inversion(pdvec, grid):
    """
    sample from a probability distribution vector, according to a grid of values
    """
    if not np.allclose(pdvec.sum(), 1):
        pdvec = pdvec / pdvec.sum()
    cdvec = np.cumsum(pdvec)
    np.testing.assert_allclose(cdvec[-1], 1)
    while True:
        rval = np.random.random()
        topidx = np.sum(cdvec <= rval) -1
        if topidx >= 0:
            return grid[topidx]

def metropolis(state, current, proposal, logp, configs):
    current_logp = logp(state, current)
    new_val = proposal.rvs(loc=current, scale=configs.jump)
    new_logp = logp(state, new_val)
    forwards = proposal.logpdf(new_val, loc=current, scale=configs.jump)
    backward = proposal.logpdf(current, loc=new_val, scale=configs.jump)
    
    hastings_factor = backwards - forwards
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
