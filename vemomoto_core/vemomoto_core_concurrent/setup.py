'''
Setup of the package vemomoto_core.vemomoto_core_concurrent
'''

from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

PATHADD = 'vemomoto_core/concurrent/'
PACKAGEADD = PATHADD.replace("/", ".")

setup(
    name="vemomoto_core_concurrent",
    version="0.9.0.a1",
    install_requires=['numpy', 'sharedmem'], 
    packages=[PACKAGEADD[:-1]],
    python_requires='>=3.6',

    # metadata to display on PyPI
    author="Samuel M. Fischer",
    license='LGPLv3',
    description="Methods to ease parallelization of code", 
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords="parallel, shared memory, multiprocessing, concurrency", 
    url="https://github.com/vemomoto/vemomoto",
    project_urls={
        "Bug Tracker": "https://github.com/vemomoto/vemomoto/issues",
        "Documentation": "https://vemomoto.github.io/vemomoto_core/vemomoto_core.concurrent",
        "Source Code": "https://github.com/vemomoto/vemomoto/tree/master/vemomoto_core/vemomoto_core_concurrent",
    },
    classifiers=[
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
    ],
)
