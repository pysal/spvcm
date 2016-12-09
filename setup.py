from setuptools import setup

setup(name='hlm_gibbs',
      version='0.0.1',
      description='Fit spatial multilevel models and diagnose convergence',
      url='https://github.com/ljwolf/hlm_gibbs',
      author='Levi John Wolf',
      author_email='levi.john.wolf@gmail.com',
      license='3-Clause BSD',
      packages=['hlm_gibbs'],
      install_requires=['numpy','scipy','pysal','pandas','seaborn']
      zip_safe=False)
