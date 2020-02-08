'''
Created on 14.02.2017

@author: Samuel
'''
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

NAME = 'intquickheapdict'
extensions = [Extension(NAME, [NAME+'.pyx'],
                        extra_compile_args=['-std=c++11', '-O3'],
                        include_dirs=[np.get_include()],
                        )
              ]

setup(name=NAME,
      ext_modules = cythonize(extensions, language="c++"),
      )
