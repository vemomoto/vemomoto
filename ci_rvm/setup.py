'''
Created on 28.12.2019

@author: Samuel
'''

from setuptools import setup


PATHADD = 'ci_rvm/'
PACKAGEADD = PATHADD.replace("/", ".")

setup(
    name="ci_rvm",
    version="0.9.0.dev1",
    #packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib', 'vemomoto_core_tools'], 
    packages=[PACKAGEADD[:-1]],
    package_data={
        # If any package contains *.rst files, include them:
        '': ['*.rst'],
    },

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
