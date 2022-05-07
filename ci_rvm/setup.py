'''
Setup of the package ci_rvm
'''

from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

PATHADD = 'ci_rvm/'
PACKAGEADD = PATHADD.replace("/", ".")

setup(
    name="ci_rvm",
    version="0.9.1.a2",
    install_requires=['numpy', 'scipy', 'matplotlib', 'vemomoto_core_tools'], 
    packages=[PACKAGEADD[:-1]],
    package_data={
        '': ['*LICENSE*'],
    },
    license='LGPLv3',
    python_requires='>=3.6',

    # metadata to display on PyPI
    author="Samuel M. Fischer",
    description="An algorithm to find profile likelihood confidence intervals", 
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords="confidence interval, likelihood, profile likelihood", 
    url="https://github.com/vemomoto/vemomoto",
    project_urls={
        "Bug Tracker": "https://github.com/vemomoto/vemomoto/issues",
        "Documentation": "https://vemomoto.github.io/ci_rvm",
        "Source Code": "https://github.com/vemomoto/vemomoto/tree/master/ci_rvm",
        "Publication": "https://link.springer.com/article/10.1007/s11222-021-10012-y",
    },
    classifiers=[
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
    ],
)
