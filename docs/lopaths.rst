Package: lopaths
================


.. toctree::

   graph <lopaths/lopaths.graph>
   graph_utils <lopaths/lopaths.graph_utils>
   sig_fig_rounding <lopaths/lopaths.sig_fig_rounding>
   test_graph <lopaths/lopaths.test_graph>
   test_routes <lopaths/lopaths.test_routes>


Installation
------------------------------------------------

The package can be installed via `pip <https://pypi.org/project/pip/>`_. To install the package, you can use 

.. code-block::

	pip install lopaths
	
Please note that a compiler may be needed to install the package. Please refer to the section `Installation <index.html#installation>`_ on the main page.

.. note:: Some algorithms implemented in this package rely on share memory libraries that work on Unix systems only. If locally optimal paths shall be computed in large road networks, it is strongly encouraged to execute the code on Linux, as some tasks are not implemented to run in parallel on Windows.


Scientific Publication
------------------------------------------------

The algorithms implemented in this package are explained in the paper "`Locally optimal routes for route choice sets <https://arxiv.org/abs/1909.08801>`_" (preprint). Please cite this publication if you have used the package in your own research.