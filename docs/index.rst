.. hybrid_vector_model documentation master file, created by
   sphinx-quickstart on Wed Jan 29 20:48:11 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VeMoMoTo - Vector Movement Modelling Tools 
==========================================

"Vector movement modelling tools" is a collection of python packages aiming to model the movement of invasive species or disease vectors through road networks. 

- The package :doc:`hybrid_vector_model <hybrid_vector_model>` provides the core functionality for the model. 

- The package :doc:`lopaths <lopaths>` contains an algorithm to identify locally optimal routes in road networks. These routes are in particular a necessary requirement for the hybrid vector model. 

- The package :doc:`ci_rvm <ci_rvm>` contains an algorithm to identify profile likelihood confidence intervals. This algorithm is in particular used to assess the hybrid vector model

- The package :doc:`vemomoto_core <vemomoto_core>` provides helpful tools required by the other packages

.. warning:: This documentation and also parts of the software packages are still under construction. The API is not yet complete, and working examples are still to be addedd. Furthermore, some variables and methods may be renamed in future versions without notice, so be cautious with using this early version of the software. 

Installation
-------------------------------------------------

The packages are available on the `Python package index <https://pypi.org/>`_ and can be installed via `pip <https://pypi.org/project/pip/>`_. The project includes some C extensions, which need to be compiled before they can be used. For 64-bit Windows, precompiled packages are available. If working on a different platform, make sure that a compiler is installed. If working with `Anaconda <https://www.anaconda.com/>`_, you may run  

.. code-block::
	
	conda install gxx_linux-64
	conda install cmake
	
If working on a 32-bit machine, replace the ``64`` with a ``32``. The packages have not been tested on a Mac yet, but replacing ``gxx_linux-64`` with ``clangxx_osx-64`` should work. See the `Anaconda documentation <https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html>`_ for details.


To install all packages, you may use 

.. code-block::

	pip install vemomoto

Installation instructions for individual packages can be found on the corresponding subpages.

The optimizier for vector control included in this package depends on the commercial software `MOSEK <https://www.mosek.com/>`_. If this software is not installed, a respective warning may be displayed upon import of the packages. For instructions on how to install MOSEK, and for the respective terms and conditions, refer to the `MOSEK documentation <https://docs.mosek.com/9.1/pythonfusion/install-interface.html>`_. If working with Anaconda, you may use 

.. code-block::

	conda install -c mosek mosek

.. note:: Some algorithms implemented in this package rely on share memory libraries that work on Unix systems only. If the vemomoto packages are to be applied to large sytems, it is strongly encouraged to execute the code on Linux, as some tasks are not implemented to run in parallel on Windows.

License and Referencing in Scientific Publications
--------------------------------------------------

This project is distributed under the the `LGPL v3 license <https://opensource.org/licenses/lgpl-3.0.html>`_. Some of the packages are based on scientific publications. If you publish results obtained with the help of one of the packages in a scientific journal, please cite the corresponding papers. Links to the corresponding papers can be found on the respective subpages.


Bugs, Feature Requests, and Contributions
--------------------------------------------------
If you have found a bug or have a feature request, you may create a ticket on the project's `github page <https://github.com/vemomoto/vemomoto>`_. You are also invited to fork and contribute to the project!



API - Table of Contents
--------------------------------------------------

.. toctree::
   :maxdepth: 4
   :titlesonly:
   
   hybrid_vector_model <hybrid_vector_model>
   ci_rvm <ci_rvm>
   lopaths <lopaths>
   vemomoto_core <vemomoto_core>


Indices and Tables
--------------------------------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
