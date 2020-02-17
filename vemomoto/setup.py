'''
Setup of the package collection vemomoto
'''

from setuptools import setup



setup(
    name="vemomoto",
    version="0.9.0.dev1",
    #packages=find_packages(),
    package_data={
        '': ['*LICENSE*'],
    },
    
    install_requires=["hybrid_vector_model", "lopaths", "ci_rvm"],
    
    # metadata to display on PyPI
    author="Samuel M. Fischer",
    description="A collection of python packages aiming to model the movement of invasive species or disease vectors through road networks", 
    url="https://github.com/vemomoto/vemomoto",
    project_urls={
        "Bug Tracker": "https://github.com/vemomoto/vemomoto",
        "Documentation": "https://vemomoto.github.io",
        "Source Code": "https://github.com/vemomoto/vemomoto",
    },
    classifiers=[
        'License :: OSI Approved :: LGPL-3.0'
    ],
)
