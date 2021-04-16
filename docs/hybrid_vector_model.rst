Package: hybrid\_vector\_model
==============================

.. toctree::
	:titlesonly:
	
	hybrid_vector_model <hybrid_vector_model/hybrid_vector_model.hybrid_vector_model>
	route_choice_model <hybrid_vector_model/hybrid_vector_model.route_choice_model>
	statsutils <hybrid_vector_model/hybrid_vector_model.statsutils>
	traveltime_model <hybrid_vector_model/hybrid_vector_model.traveltime_model>
	boater_movement_model <hybrid_vector_model/hybrid_vector_model.boater_movement_model>


Installation
------------

The package can be installed via `pip <https://pypi.org/project/pip/>`_. To install the package, you can use 

.. code-block::

	pip install hybrid_vector_model
	
Please note that a compiler may be needed to install the package. Please refer to the section `Installation <index.html#installation>`_ on the main page for details.

The optimizier for vector control included in this package depends on the commercial software `MOSEK <https://www.mosek.com/>`_. See the section `Installation <index.html#installation>`_ on the main page for details and installation instructions.

.. note:: Some algorithms implemented in this package rely on share memory libraries that work on Unix systems only. If the hybrid vector model shall be applied to large sytems, it is strongly encouraged to execute the code on Linux, as some tasks are not implemented to run in parallel on Windows.

Usage
-----

The hybrid vector model is implemented in the `HybridVectorModel <hybrid_vector_model/hybrid_vector_model.hybrid_vector_model.html#hybrid_vector_model.hybrid_vector_model.HybridVectorModel>`_ class. 
To create a hybrid model for a system of interest, it is convenient to use the `HybridVectorModel.new(...) <hybrid_vector_model/hybrid_vector_model.hybrid_vector_model.html#hybrid_vector_model.hybrid_vector_model.HybridVectorModel.new>`_ method, 
which takes several data files as input and creates and fits a hybrid vector model. The model is then returned.

Since the covariates used to model traffic incentive (repulsiveness of donors and attractiveness of recipients) may vary from system to system,
it is necessary to provide a class representing the travel incentive model (here called 'traffic factor model') to the ``new`` method. 
The class for the traffic factor model must inherit from `BaseTrafficFactorModel <hybrid_vector_model/hybrid_vector_model.hybrid_vector_model.html#hybrid_vector_model.hybrid_vector_model.BaseTrafficFactorModel>`_, and provides
a method returning a factor proportional to the traffic between each donor and recipient and a list of all the required covariates.

In this package, an example for such a traffic factor model is implemented with respect to boater traffic from jurisdictions to lakes. The 
implemented traffic factor model  
can be found in the module `boater_movement_model <hybrid_vector_model/hybrid_vector_model.boater_movement_model.html#hybrid_vector_model.boater_movement_model.TrafficFactorModel>`_, and may be helpful as a reference
when building a custom model.

In conclusion, a model could be created and fitted as follows:

.. code-block:: python
	
	import os
	
	# Import the class implementing the hybrid vector
	# model
	from hybrid_vector_model import HybridVectorModel

	# Import the class implementing the traffic factor model
	# Instead of this import, it may be better to implement
	# your own class tailored to your system.
	from boater_movement_model import TrafficFactorModel

	# Reuse earlier results if possible
	restart = False
	
	# Declare the file names. Because we assume here that the 
	# files are in a subdirectory 'Example', we need to merge 
	# the file names accordingly.
	# See the documentation for HybridVectorModel.new for a 
	# detailed description of the files and their contents.
	folder = "Example"
	fileNameEdges = os.path.join(folder, "Edges.csv")
	fileNameVertices = os.path.join(folder, "Vertices.csv")
	fileNameOrigins = os.path.join(folder, "PopulationData.csv")
	fileNameDestinations = os.path.join(folder, "LakeData.csv")
	fileNamePostalCodeAreas = os.path.join(folder, "PostalCodeAreas.csv")
	fileNameObservations = os.path.join(folder, "SurveyData.csv")

	# Set the compliance rate of travellers. This is the fraction of
	# travellers who would stop at a survey location and comply with a survey.
	# Typically, this rate cannot be computed directly from 
	# survey data and must therefore be specified independently.
	complianceRate = 0.8

	# File name of the model
	fileNameSave = "Example"

	# These parameters define which routes are deemed likely.
	# The first parameter is the factor by how much an admissible
	# route may be longer than the shortest route. 
	# The second parameter specifies the length of subpaths of the 
	# route that are required to be optimal (length given as fraction 
	# of the total length). 0: no restrictions, 1: only optimal paths
	# are considered. 
	# The last two parameters control internal approximations. Choosing 
	# 1 in both cases yields exact results.
	routeParameters = (1.4, .2, 1, 1)

	# create and fit a hybrid traffic model
	model = HybridVectorModel.new(
				fileNameBackup=fileNameSave, 
				trafficFactorModel_class=TrafficFactorModel,
				fileNameEdges=fileNameEdges,
				fileNameVertices=fileNameVertices,
				fileNameOrigins=fileNameOrigins,
				fileNameDestinations=fileNameDestinations,
				fileNamePostalCodeAreas=fileNamePostalCodeAreas,
				fileNameObservations=fileNameObservations,
				complianceRate=complianceRate,
				routeParameters=routeParameters, 
				restart=restart
				)

Refer to `HybridVectorModel.new(...) <hybrid_vector_model/hybrid_vector_model.hybrid_vector_model.html#hybrid_vector_model.hybrid_vector_model.HybridVectorModel.new>`_
for a description of the data files required. The example is also implemented in the method 
`boater_movement_model.example() <hybrid_vector_model/hybrid_vector_model.boater_movement_model.html#hybrid_vector_model.boater_movement_model.example>`_.

Example data files are provided with the package in the subfolder ``hybrid_vector_model/Example``. Alternatively, these files 
can be downloaded from the `github repository <https://github.com/vemomoto/vemomoto/tree/master/hybrid_vector_model/hybrid_vector_model/Example>`_. 
There you can also find a graphical image of the example road network.


Scientific Publications
-----------------------

The theory behind the model implemented in this package is explained in the paper "`A hybrid gravity and route choice model to assess vector traffic in large-scale road networks <https://doi.org/10.1098/rsos.191858>`_". 
The algorithm for optimizing inspection stations is described in "`Managing aquatic invasions: Optimal locations and operating times for watercraft inspection stations <https://doi.org/10.1016/j.jenvman.2020.111923>`_".
Please cite the corresponding publication(s) if you have used the package in your own research.

