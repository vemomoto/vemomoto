'''
Created on 14.02.2017

@author: Samuel
'''
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

profile = False

if profile:
    print("Compile with profiling options!")
    define_macros = [('CYTHON_TRACE', '1')]
    compiler_directives={'linetrace': True, 'binding':True}
    extra_compile_args = []
else:
    define_macros = []
    compiler_directives={'boundscheck':False, 'wraparound':False, "nonecheck":False}
    extra_compile_args = ['-O3']
    #compiler_directives={}

NAME = 'FixedOrderedIntDict'
extensions = [Extension(NAME, [NAME+'.pyx'],
                        include_dirs=[np.get_include()],
                        define_macros=define_macros #Profiling
                        )
              ]

setup(name=NAME,
      ext_modules = cythonize(extensions,  
                              compiler_directives=compiler_directives),
      include_dirs=[np.get_include()]
      )
