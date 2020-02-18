'''
Setup of the package collection vemomoto
'''

from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name="vemomoto",
    version="0.9.0.a1",
    #packages=find_packages(),
    package_data={
        '': ['*LICENSE*'],
    },
    
    install_requires=["hybrid_vector_model", "lopaths", "ci_rvm"],
    python_requires='>=3.6',
    
    # metadata to display on PyPI
    license='LGPLv3',
    author="Samuel M. Fischer",
    description="A collection of python packages aiming to model the movement of invasive species or disease vectors through road networks", 
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/vemomoto/vemomoto",
    project_urls={
        "Bug Tracker": "https://github.com/vemomoto/vemomoto/issues",
        "Documentation": "https://vemomoto.github.io",
        "Source Code": "https://github.com/vemomoto/vemomoto",
    },
    classifiers=[
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
    ],
)
