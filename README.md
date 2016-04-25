This is the refactored HSAR sampler code, now ready to start adding on new user
classes.

The specification for a any Userclass/Baseclass pair that needs to be estimated
via gibbs sampling is in `./hlm/spec.md`. 

An example of estimating the HSAR and comparing the results to an OLS regression
are shown in `./hsar_userclass.ipynb`. 

As more models are added, they will be given their own subdirectory in `./hlm`. 
