from setuptools import setup, find_packages

import os

fh = os.path.join(os.path.dirname(__file__), 'README.rst')
with open(fh, 'r') as f:
    long_description = f.read()

# Get __version__ from spvcm/__init__.py without importing the package
# __version__ has to be defined in the first line
with open('spvcm/__init__.py', 'r') as f:
    exec(f.readline())


setup(name='spvcm',
      version=__version__,
      long_description = long_description,
      description='Fit spatial multilevel models and diagnose convergence',
      url='https://github.com/ljwolf/spvcm',
      author='Levi John Wolf',
      author_email='levi.john.wolf@gmail.com',
      license='3-Clause BSD',
      packages= find_packages(),
      install_requires=['numpy','scipy','libpysal', 'spreg', 'pandas','seaborn'],
      include_package_data=True,
      zip_safe=False)
