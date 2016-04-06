# python version 2.7
# Author: Siyu Zhu
# Jan 13, 2014
# python cython distutils
# filename: setup.py
# ==================================================================================
from distutils.core import setup
from Cython.Build import cythonize

setup(
	name = 'ckmeans module',
	ext_modules = cythonize('ckmean.pyx'),
)
