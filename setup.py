from setuptools import setup, find_packages

setup(name='mlm_gibbs',
      version='0.0.1',
      description='Fit spatial multilevel models and diagnose convergence',
      url='https://github.com/ljwolf/mlm_gibbs',
      author='Levi John Wolf',
      author_email='levi.john.wolf@gmail.com',
      license='3-Clause BSD',
      packages= find_packages(),
      install_requires=['numpy','scipy','pysal','pandas','seaborn'],
      include_package_data=True,
      zip_safe=False)
