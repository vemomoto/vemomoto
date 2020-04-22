'''
Setup of the package hybrid_vector_model
'''

from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

PATHADD = 'hybrid_vector_model/'
PACKAGEADD = PATHADD.replace("/", ".")

setup(
    name="hybrid_vector_model",
    version="0.9.0.b6",
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas', 'statsmodels', 
                      'cvxpy', 'autograd', 'vemomoto_core_npcollections',
                      'numdifftools', 'vemomoto_core_tools>=0.9.0.a4',
                      'vemomoto_core_concurrent', 'lopaths>=0.9.0.a5', 'ci_rvm'], 
    packages=[PACKAGEADD[:-1]],
    package_data={
        '': ['*LICENSE*', '*Example/*', '*cvxpy_changes/*'],
    },
    python_requires='>=3.6',

    # metadata to display on PyPI
    author="Samuel M. Fischer",
    license='LGPLv3',
    description="Package to model the traffic of invasive species and disease vectors", 
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords="gravity model, hierarchical model, infectious disease, invasive species, propagule pressure", 
    url="https://github.com/vemomoto/vemomoto",
    project_urls={
        "Bug Tracker": "https://github.com/vemomoto/vemomoto/issues",
        "Documentation": "https://vemomoto.github.io/hybrid_vector_model",
        "Source Code": "https://github.com/vemomoto/vemomoto/tree/master/hybrid_vector_model",
        "Publication": "https://arxiv.org/abs/1909.08811",
    },
    classifiers=[
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
    ],
    extras_require={
        'inspection_optimization':  ["mosek"],
    }
)
