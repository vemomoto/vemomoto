'''
Created on 28.12.2019

@author: Samuel
'''

from setuptools import setup


PATHADD = 'vemomoto_core/tools/'
PACKAGEADD = PATHADD.replace("/", ".")

setup(
    name="vemomoto_core_tools",
    version="0.9.0.dev1",
    #packages=find_packages(),
    install_requires=['dill'], 
    packages=[PACKAGEADD[:-1]],

    # metadata to display on PyPI
    author="Samuel M. Fischer",
    description="Tools simplifying iteration, printing, saving, profiling, and documenting", 
    keywords="documentation, print, iterate, save, profile", 
    url="https://github.com/vemomoto/vemomoto",
    project_urls={
        "Bug Tracker": "https://github.com/vemomoto/vemomoto",
        "Documentation": "https://vemomoto.github.io/vemomoto_core/vemomoto_core.tools",
        "Source Code": "https://github.com/vemomoto/vemomoto/tree/master/vemomoto_core/vemomoto_core_tools",
    },
    classifiers=[
        'License :: OSI Approved :: LGPL-3.0'
    ],
