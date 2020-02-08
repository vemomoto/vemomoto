'''
Created on 14.02.2017

@author: Samuel
'''
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

NAME = 'npextc'

if os.name == "posix":
    parlinkargs = ['-fopenmp']
    parcompileargs = ['-fopenmp']
else: 
    parlinkargs = ['/openmp']
    parcompileargs = ['/openmp']

extensions = [Extension(NAME, [NAME+'.pyx'],
                        extra_compile_args=['-std=c++11', '-O3']+parcompileargs,
                        extra_link_args=parlinkargs,
                        include_dirs=[np.get_include()],
                        )
              ]

setup(name=NAME,
      ext_modules = cythonize(extensions, language="c++"),
      )
