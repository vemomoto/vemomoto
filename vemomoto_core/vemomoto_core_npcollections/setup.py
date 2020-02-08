'''
Created on 28.12.2019

@author: Samuel
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
    #packages=find_packages(),
    cmdclass={'build_ext' : my_build_ext},
    setup_requires=['numpy'],
    install_requires=['numpy', 'scipy', 'vemomoto_core_tools'], 
    packages=['vemomoto_core', PACKAGEADD[:-1]],
    ext_modules=extensions,
    package_data={
        '': ['*.rst', '*.pxd', '*.pyx', '*.c', '*.cpp'],
    },
    zip_safe=False,
    # metadata to display on PyPI
    author="Samuel M. Fischer",
    author_email="samuel.fischer@ualberta.ca",
    #description="", #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #keywords="", #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #url="",   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #project_urls={
    #    "Bug Tracker": "https://bugs.example.com/HelloWorld/",
    #    "Documentation": "https://docs.example.com/HelloWorld/",
    #    "Source Code": "https://code.example.com/HelloWorld/",
    #},
    #classifiers=[
    #    'License :: OSI Approved :: Python Software Foundation License'
    #]
    #    extras_require={
    #    'PDF':  ["ReportLab>=1.2", "RXP"],
    #    'reST': ["docutils>=0.3"],
    #}
)
