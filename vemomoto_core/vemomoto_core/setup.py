'''
Setup of the package collection vemomoto_core
'''

from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

PATHADD = 'vemomoto_core/'
PACKAGEADD = PATHADD.replace("/", ".")

setup(
    name="vemomoto_core",
    version="0.9.0.a1",
    install_requires=[
        'vemomoto_core_npcollections',
        'vemomoto_core_concurrent',
        'vemomoto_core_tools'], 
    python_requires='>=3.6',

    # metadata to display on PyPI
    author="Samuel M. Fischer",
    license='LGPLv3',
    description="Packages providing base functionality used by many packages in the vemomoto collection", 
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/vemomoto/vemomoto",
    project_urls={
        "Bug Tracker": "https://github.com/vemomoto/vemomoto/issues",
        "Documentation": "https://vemomoto.github.io/vemomoto_core",
        "Source Code": "https://github.com/vemomoto/vemomoto/tree/master/vemomoto_core",
    },
    classifiers=[
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
    ],
)
