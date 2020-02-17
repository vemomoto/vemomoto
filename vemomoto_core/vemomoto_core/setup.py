'''
Setup of the package collection vemomoto_core
'''

from setuptools import setup


PATHADD = 'vemomoto_core/'
PACKAGEADD = PATHADD.replace("/", ".")

setup(
    name="vemomoto_core",
    version="0.9.0.dev1",
    install_requires=[
        'vemomoto_core_npcollections',
        'vemomoto_core_concurrent',
        'vemomoto_core_tools'], 

    # metadata to display on PyPI
    author="Samuel M. Fischer",
    description="Packages providing base functionality used by many packages in the vemomoto collection", 
    url="https://github.com/vemomoto/vemomoto",
    project_urls={
        "Bug Tracker": "https://github.com/vemomoto/vemomoto",
        "Documentation": "https://vemomoto.github.io/vemomoto_core",
        "Source Code": "https://github.com/vemomoto/vemomoto/tree/master/vemomoto_core",
    },
    classifiers=[
        'License :: OSI Approved :: LGPL-3.0'
    ],
)
