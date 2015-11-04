pyHSAR
========
*edit 11/4/2015: Whoops, realized I was pushing to my github version of this.
Work should be current now*.

This is @ljwolf's work on estimating an HSAR in python. 

I still need to:

1. implement full conditionals for
    - lambda
    - rho
2. run the gibbs sampler to validate the results. 

I know the samplers for everything aside from lambda and rho work as expected.
Right now, I'm just using a dummy Unif(0,1) for rho and lambda, to see if the
gibbs sampler will even step through correctly. 

Code
=====

- `trace.py` - mocks PyMC3 traces. deliberately designed to be easy to port to
PyMC3 when ready. 
- `samplers.py` - Gibbs sampler & sampling steps. The `Gibbs` class would be akin
to the PyMC3 `Metropolis`, `Slice`, or `NUTS` samplers. I might eventually push
this back upstream. 
- `run_HSAR.py` - builds the data and sets up the sampler. This is problem
specific. The stuff I'm putting in `trace` and `samplers` is what could get
reused, and is the direct port of the HSAR.R code.  

Data
=====

- `test.csv` - taken from the dropbox folder `code/dong_harris_hsar`
- `w_lower.mtx` - taken from the dropbox folder above
- `w_upper.mtx` - taken from the dropbox folder above
`
