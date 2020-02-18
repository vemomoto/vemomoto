'''
Setup of the package vemomoto_core.vemomoto_core_tools
'''

from setuptools import setup


PATHADD = 'vemomoto_core/tools/'
PACKAGEADD = PATHADD.replace("/", ".")

setup(
    name="vemomoto_core_tools",
    version="0.9.0.a1",
    install_requires=['dill'], 
    packages=[PACKAGEADD[:-1]],
    python_requires='>=3.6',

    # metadata to display on PyPI
    author="Samuel M. Fischer",
    license='LGPL-3.0',
    description="Tools simplifying iteration, printing, saving, profiling, and documenting", 
    keywords="documentation, print, iterate, save, profile", 
    url="https://github.com/vemomoto/vemomoto",
    project_urls={
        "Bug Tracker": "https://github.com/vemomoto/vemomoto",
        "Documentation": "https://vemomoto.github.io/vemomoto_core/vemomoto_core.tools",
        "Source Code": "https://github.com/vemomoto/vemomoto/tree/master/vemomoto_core/vemomoto_core_tools",
    },
    classifiers=[
        'License :: OSI Approved :: LGPL-3.0',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
    ],
)