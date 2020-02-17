'''
Setup of the package hybrid_vector_model
'''

from setuptools import setup


PATHADD = 'hybrid_vector_model/'
PACKAGEADD = PATHADD.replace("/", ".")

setup(
    name="hybrid_vector_model",
    version="0.9.0.dev1",
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas', 'statsmodels', 
                      'cvxpy', 'autograd', 'vemomoto_core_npcollections',
                      'vemomoto_core_tools',
                      'vemomoto_core_concurrent', 'lopaths', 'ci_rvm'], 
    packages=[PACKAGEADD[:-1]],
    package_data={
        '': ['*LICENSE*', '*Example/*', '*cvxpy_changes/*'],
    },

    # metadata to display on PyPI
    author="Samuel M. Fischer",
    description="Package to model the traffic of invasive species and disease vectors", 
    keywords="gravity model, hierarchical model, infectious disease, invasive species, propagule pressure", 
    url="https://github.com/vemomoto/vemomoto",
    project_urls={
        "Bug Tracker": "https://github.com/vemomoto/vemomoto",
        "Documentation": "https://vemomoto.github.io/hybrid_vector_model",
        "Source Code": "https://github.com/vemomoto/vemomoto/tree/master/hybrid_vector_model",
        "Publication": "https://arxiv.org/abs/1909.08811",
    },
    classifiers=[
        'License :: OSI Approved :: LGPL-3.0'
    ],
    extras_require={
        'inspection_optimization':  ["mosek"],
    }
)
