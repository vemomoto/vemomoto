'''
Setup of the package vemomoto_core.npcollections
'''
import os
from setuptools import setup, Extension

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    

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
            
            from Cython.Build import cythonize
            self.distribution.ext_modules = cythonize(self.distribution.ext_modules,
                                                      force=True)

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

ext = '.pyx'
    
extensions = [Extension(PACKAGEADD+name, [PATHADD+name+ext],
                        extra_compile_args=['-std=c++11', '-O3']+parcompileargs,
                        extra_link_args=parlinkargs,
                        )
              for name in extnames]
for e in extensions:
    e.language_level = 3
    e.language = 'c++'
    
setup(
    name="vemomoto_core_npcollections",
    version="0.9.0.a11",
    cmdclass={'build_ext' : my_build_ext},
    setup_requires=['numpy', "cython"],
    install_requires=['numpy', 'scipy', 'vemomoto_core_tools', "cython"], 
    python_requires='>=3.6',
    packages=['vemomoto_core', PACKAGEADD[:-1]],
    ext_modules=extensions,
    package_data={
        '': ['*.pxd', '*.pyx', 'heapoperations_c.c', 'sectionsum.c'],
    },
    zip_safe=False,
    
    # metadata to display on PyPI
    author="Samuel M. Fischer",
    license='LGPLv3',
    description="Flexible containers based on numpy arrays", 
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords="numpy, array, container, heap, dictionary", 
    url="https://github.com/vemomoto/vemomoto",
    project_urls={
        "Bug Tracker": "https://github.com/vemomoto/vemomoto/issues",
        "Documentation": "https://vemomoto.github.io/vemomoto_core/vemomoto_core.npcollections",
        "Source Code": "https://github.com/vemomoto/vemomoto/tree/master/vemomoto_core/vemomoto_core_npcollections",
    },
    classifiers=[
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
    ],
)
