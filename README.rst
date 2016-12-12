===========================================================================
``spvcm``: Gibbs sampling for spatially-correlated variance-components
===========================================================================

This is a package to estimate spatially-correlated variance components models, 
an equivalent Bayesian model formulation to Varying-Intercept models. 
*author*: Levi John Wolf
*email*: ``levi.john.wolf@gmail.com``
*institution*: Arizona State University & University of Chicago Center for Spatial Data Science

--------------------
Installation
--------------------

This package works best in Python 3.5, but unittests pass in Python 2.7 as well. 
Only Python 3.5+ is officially supported. 

To install, first install the Anaconda Python Distribution_ from Continuum Analytics_. Installation of the package has been tested in Windows (10, 8, 7) Mac OSX (10.8+) and Linux using Anaconda 4.2.0, with Python version 3.5. 

Once Anaconda is installed, ``spvcm`` can be installed using ``pip``, the Python Package Manager. 

``pip install spvcm``

To install this from source, one can also navigate to the source directory and use:

``pip install ./``

which will install the package from the target source directory. 

-------------------
Usage
-------------------

To use the package, start up a Python interpreter and run:
``import spvcm.api as spvcm``

Then, many differnet variance components model specificaions are available in:

``spvcm.both``
``spvcm.upper``
``spvcm.lower``

For more thorough directions, consult the Jupyter Notebook, ``using the sampler.ipynb``, which is attached with the package. 

-------------------
Citation
-------------------

Levi John Wolf. (2016). `Gibbs Sampling for a class of  spatially-correlated variance components models`. University of Chicago Center for Spatial Data Science Technical Report. 

.. _Distribution: https://https://www.continuum.io/downloads
.. _Analytics: https://continuum.io
.. _package: 