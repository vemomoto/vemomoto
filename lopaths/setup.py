'''
Setup of the package lopaths
'''
import os
from setuptools import setup
from setuptools.extension import Extension

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
            import vemomoto_core.npcollections as npcollections
            self.include_dirs.append(os.path.dirname(npcollections.__file__))

    #object returned:
    return build_ext(pars)

# cython c++ extensions
extnames = [
    'graph_utils',
    ]

if os.name == "posix":
    parlinkargs = ['-fopenmp']
    parcompileargs = ['-fopenmp']
else: 
    parlinkargs = ['/openmp']
    parcompileargs = ['/openmp']

PATHADD = 'lopaths/'
PACKAGEADD = PATHADD.replace("/", ".")

extensions = [Extension(PACKAGEADD+name, [PATHADD+name+'.cpp'],
                        extra_compile_args=['-std=c++11', '-O3']+parcompileargs,
                        extra_link_args=parlinkargs,
                        )
              for name in extnames]

setup(
    name="lopaths",
    version="0.9.0.a6",
    cmdclass={'build_ext' : my_build_ext},
    setup_requires=['numpy', 'vemomoto_core_npcollections'],
    install_requires=[
        'numpy', 
        'sharedmem', 
        'vemomoto_core_concurrent',
        'vemomoto_core_npcollections',
        'vemomoto_core_tools>=0.9.0.a4'
        ], 
    python_requires='>=3.6',
    packages=[PACKAGEADD[:-1]],
    ext_modules=extensions,
    package_data={
        '': ['*.pxd', '*.pyx', '*.c', '*.cpp'],
    },
    zip_safe=False,
    
    # metadata to display on PyPI
    license='LGPLv3',
    author="Samuel M. Fischer",
    description="Package to find locally optimal routes in route networks", 
    keywords="alternative paths, choice set, local optimality, road network, route choice", 
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/vemomoto/vemomoto",
    project_urls={
        "Bug Tracker": "https://github.com/vemomoto/vemomoto/issues",
        "Documentation": "https://vemomoto.github.io/lopaths",
        "Source Code": "https://github.com/vemomoto/vemomoto/tree/master/lopaths",
        "Publication": "https://arxiv.org/abs/1909.08801",
    },
    classifiers=[
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
    ],
    extras_require={
        'cython_compilation':  ["cython"],
    }
)
