'''
Setup of the package vemomoto_core.vemomoto_core_tools
'''

from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

PATHADD = 'vemomoto_core/tools/'
PACKAGEADD = PATHADD.replace("/", ".")


setup(
    name="vemomoto_core_tools",
    version="0.9.0.b3",
    install_requires=['dill'], 
    packages=[PACKAGEADD[:-1]],
    python_requires='>=3.6',

    # metadata to display on PyPI
    author="Samuel M. Fischer",
    license='LGPLv3',
    description="Tools simplifying iteration, printing, saving, profiling, and documenting", 
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords="documentation, print, iterate, save, profile", 
    url="https://github.com/vemomoto/vemomoto",
    project_urls={
        "Bug Tracker": "https://github.com/vemomoto/vemomoto/issues",
        "Documentation": "https://vemomoto.github.io/vemomoto_core/vemomoto_core.tools",
        "Source Code": "https://github.com/vemomoto/vemomoto/tree/master/vemomoto_core/vemomoto_core_tools",
    },
    classifiers=[
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
    ],
)