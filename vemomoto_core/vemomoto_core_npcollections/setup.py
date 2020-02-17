'''
Setup of the package vemomoto_core.npcollections
'''
import os
from setuptools import setup
from setuptools.extension import Extension

# factory function
def my_build_ext(pars):
    # import delayed:
    from setuptools.command.build_ext import build_ext as _build_ext
    
    # include_dirs adjusted: 
    class build_ext(_build_ext):
        def finalize_options(self):
            _build_ext.finalize_options(self)
            # Prevent numpy from thinking it is still in its setup process:
            __builtins__.__NUMPY_SETUP__ = False
            import numpy
            self.include_dirs.append(numpy.get_include())

    #object returned:
    return build_ext(pars)

# cython c++ extensions
extnames = [
    'npextc',
    'intquickheapdict',
    'FixedOrderedIntDict'
    ]

if os.name == "posix":
    parlinkargs = ['-fopenmp']
    parcompileargs = ['-fopenmp']
else: 
    parlinkargs = ['/openmp']
    parcompileargs = ['/openmp']

PATHADD = 'vemomoto_core/npcollections/'
PACKAGEADD = PATHADD.replace("/", ".")

extensions = [Extension(PACKAGEADD+name, [PATHADD+name+'.cpp'],
                        extra_compile_args=['-std=c++11', '-O3']+parcompileargs,
                        extra_link_args=parlinkargs,
                        )
              for name in extnames]

setup(
    name="vemomoto_core_npcollections",
    version="0.9.0.dev1",
    cmdclass={'build_ext' : my_build_ext},
    setup_requires=['numpy'],
    install_requires=['numpy', 'scipy', 'vemomoto_core_tools'], 
    packages=['vemomoto_core', PACKAGEADD[:-1]],
    ext_modules=extensions,
    package_data={
        '': ['*.pxd', '*.pyx', '*.c', '*.cpp'],
    },
    zip_safe=False,
    # metadata to display on PyPI
    author="Samuel M. Fischer",
    description="Flexible memory containers based on numpy arrays", 
    keywords="numpy, array, container, heap, dictionary", 
    url="https://github.com/vemomoto/vemomoto",
    project_urls={
        "Bug Tracker": "https://github.com/vemomoto/vemomoto",
        "Documentation": "https://vemomoto.github.io/vemomoto_core/vemomoto_core.npcollections",
        "Source Code": "https://github.com/vemomoto/vemomoto/tree/master/vemomoto_core/vemomoto_core_npcollections",
    },
    classifiers=[
        'License :: OSI Approved :: LGPL-3.0'
    ],
    extras_require={
        'cython_compilation':  ["cython"],
    }
)
