'''
Created on 14.02.2017

@author: Samuel
'''
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

import sys
print(sys.executable)

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
    compiler_directives={}

NAME = 'graph_utils'
extensions = [Extension(NAME, [NAME+'.pyx'],
                        extra_compile_args=extra_compile_args+['-std=c++11'],
                        include_dirs=[np.get_include()],
                        define_macros=define_macros #Profiling
                        )
              ]

setup(name=NAME,
      ext_modules = cythonize(extensions, language="c++", 
                              compiler_directives=compiler_directives,
                              language_level="3"),
      include_dirs=[np.get_include()]
      ) 

