'''
Created on 28.12.2019

@author: Samuel
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
