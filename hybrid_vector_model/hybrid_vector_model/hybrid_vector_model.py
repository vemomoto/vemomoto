'''


.. note:: This program was created with the motivation to model the traffic
    of boaters potentially carrying aquatic invasive species. Nonetheless,
    the tools are applicable to assess and control any vector road traffic. 
    Due to the initial motivation, however, the wording within this file may be
    specific to the motivating scenario:
        
    - We may refer to `vectors` as `boaters` or `agents`. 
    - We may refer to the origins of vectors as `origin`, `source`, or simply `jurisdiction`. 
    - We may refer to the destinations of vectors as `destination`, `sink`, 
      or `lake`.

'''
#asd 
import os
import sys
import warnings
from copy import copy
from os.path import exists
from itertools import product as iterproduct, repeat, count as itercount, \
                        chain as iterchain
from collections import defaultdict
import traceback

import numpy as np
import numpy.lib.recfunctions as rf
from scipy import sparse
import scipy.optimize as op
from scipy.stats import nbinom, norm as normaldist, f as fdist, linregress, \
                        vonmises, chi2
import matplotlib
if os.name == 'posix':
    # if executed on a Windows server. Comment out this line, if you are working
    # on a desktop computer that is not Windows.
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import autograd.numpy as ag
from autograd import grad, hessian
from statsmodels.distributions.empirical_distribution import ECDF
import cvxpy as cp


from vemomoto_core.npcollections.npext import add_fields, list_to_csr_matrix, sparsepowersum, \
    sparsepower, sparsesum, convert_R_0_1, \
    convert_R_pos, FlexibleArrayDict
from vemomoto_core.tools import saveobject
from vemomoto_core.npcollections.npextc import FlexibleArray
from vemomoto_core.tools.hrprint import HierarchichalPrinter
from vemomoto_core.tools.tee import Tee
from vemomoto_core.tools.doc_utils import DocMetaSuperclass, add_doc
from vemomoto_core.concurrent.concurrent_futures_ext import ProcessPoolExecutor
from vemomoto_core.concurrent.nicepar import Counter
from lopaths import FlowPointGraph, FlexibleGraph
from ci_rvm import find_profile_CI_bound

try:
    from .traveltime_model import TrafficDensityVonMises
    from .statsutils import anderson_darling_test_discrete, anderson_darling_NB, anderson_darling_P, R2
    from ._autograd_ext import nbinom_logpmf
    from .route_choice_model import RouteChoiceModel
except ImportError:
    from traveltime_model import TrafficDensityVonMises
    from statsutils import anderson_darling_test_discrete, anderson_darling_NB, anderson_darling_P, R2
    from _autograd_ext import nbinom_logpmf
    from route_choice_model import RouteChoiceModel


# general settings --------------------------------------------------------
# The first command line argument specifies the output file to which all output
# will be written.                     
if len(sys.argv) > 1:
    teeObject = Tee(sys.argv[1])

np.set_printoptions(linewidth = 100)
warnings.simplefilter('always', UserWarning) 
# -------------------------------------------------------------------------

CPU_COUNT = os.cpu_count() 
"Number of processes for parallel processing."

IDTYPE = "|S11"
"""Type of IDs of origins and destinations. Alpha-numerical code with at most 9 
digets. 

2 digets remain reserved for internal use.

"""

def non_join(string1, string2):
    if string1 is not None and string2 is not None:
        return string1 + string2
    else:
        return None

class TransportNetwork(FlowPointGraph):
    """A graph representation of the road network.
    
    In contrast to the general graph class :py:class:`lopaths.graph.FlowPointGraph`,
    :py:class:`TransportNetwork` implements invasion-specific functionalities
    needed for the vector movement model
    
    Parameters
    ----------
    fileNameEdges : str 
        Name of a csv file containing the road network. The file must be a 
        csv with header and the following columns, separated by ``,``:
        
        ================================================== ============================== =========
        Field                                              Type                           Description
        ================================================== ============================== =========
        RoadID                                             :py:data:`IDTYPE`              ID of the road section
        VertexFromID                                       :py:data:`IDTYPE`              Starting vertex of the road section
        VertexToID                                         :py:data:`IDTYPE`              End vertex of the road section
        Lenght                                             float                          Length (or travel time) of the road section
        Survey location for forward traffic                :py:data:`IDTYPE`, `optional`  ID of the location
                                                                                          where forward traffic can be inspected
        Survey location for forward traffic                :py:data:`IDTYPE`, `optional`  ID of the station 
                                                                                          where backward traffic can be inspected
        Survey location for forward and backward traffic   :py:data:`IDTYPE`, `optional`  ID of the station where forward and backward traffic can be inspected
        DestinationID                                      :py:data:`IDTYPE`, `optional`  ID of the destination that can be accessed via this
                                                                                          road section
        ================================================== ============================== =========
              
    fileNameVertices : str
        Name of a csv file stating which vertices are origins and destinations.
        The file must be a csv with header and the following columns, 
        separated by ``,``:
        
        ==================== ================= =========
        Field                Type              Description
        ==================== ================= =========
        VertexID             :py:data:`IDTYPE` ID of the vertex
        potentialViaVertex   bool              whether the vertex could be a potential 
                                               intermediate destination for boaters 
                                               (should be ``True`` by default,
                                               but can be set to ``False`` for many 
                                               vertices to reduce computational complexity)
        vertexType           int               type identifier for the vertex; see below
                                                  - ``1``: origin
                                                  - ``2``: destination
                                                  - ``3``: postal code area center 
                                                    (used to determine whether destinations are 
                                                    located in populated areas)
                                                  - `other`: no specific role for the vertex
        ==================== ================= =========
                             
    destinationToDestination : bool
        ``True`` iff destination to destination traffic is modelled, i.e. if
        the sets of origins and destinations are equal.
        Note: a gravity model for destination to destination traffic is not yet 
        implemented, but could be added easily.
    preprocessingArgs : tuple
        Arguments for preprocessing of the road network. Refer to 
        :py:class:`lopaths.graph.FlowPointGraph` for further 
        documentation
    edgeLengthRandomization : float
        Maximum random perturbation to road lengths. Needed to make it 
        likely that distinct paths have distinct length.
    printerArgs
        Arguments passed to 
        :py:class:`vemomoto_core.tools.hrprint.HierarchichalPrinter`
    
    """
    
    # labels that are going to be put at the beginning of origin IDs to avoid
    # overlapping with other IDs
    _DESTINATION_SOURCE_LABEL = b'l'
    _DESTINATION_SOURCE_EDGE_LABEL = b'_l'
    _DESTINATION_SINK_LABEL = b'L'
    _DESTINATION_SINK_EDGE_LABEL = b'_L'
    
    def __init__(self, fileNameEdges, fileNameVertices, 
                 destinationToDestination=False, 
                 preprocessingArgs=None,
                 edgeLengthRandomization=.001, 
                 **printerArgs):
        """Constructor
        
        See class docs for documentation.
        
        """
        
        HierarchichalPrinter.__init__(self, **printerArgs)
        
        self.preprocessingArgs = preprocessingArgs
        self.prst("Reading road network file", fileNameEdges)
        edges = np.genfromtxt(fileNameEdges, delimiter=",",  
                                   skip_header = True, 
                                   dtype = {"names":["ID", "from_to_original", 
                                                     "length", "inspection", 
                                                     "destinationID"], 
                                            'formats':[IDTYPE, 
                                                       '2' + IDTYPE, 
                                                       "double", 
                                                       "3" + IDTYPE, 
                                                       IDTYPE]},
                              autostrip = True)
        self.prst("Reading vertex file", fileNameVertices)
        vertexData = np.genfromtxt(fileNameVertices, delimiter=",",  
                                   skip_header = True, 
                                   dtype = {"names":["ID", "potentialViaVertex",
                                                     "type"], 
                                            'formats':[IDTYPE, 'bool', 'int']},
                                   autostrip = True)
        
        if not vertexData.ndim:
            vertexData = np.expand_dims(vertexData, 0)
        
        if destinationToDestination:
            # since destination access is given directly in the road network file, 
            # we ignore the given vertexData
            vertexData["type"][vertexData["type"]==1] = -1
        
        if edgeLengthRandomization:
            np.random.seed(0)
            edges["length"] += ((np.random.random(len(edges))-0.5) 
                                * edgeLengthRandomization)
        
        self.prst("Processing the input")
        
        # double the edges to create directed graph from undirected graph
        from_to = np.vstack((edges["from_to_original"], 
                             edges["from_to_original"][:,::-1]))
        
        edgeData = rf.repack_fields(edges[["ID", "length", "destinationID"]])
        edgeData = np.concatenate((edgeData, edgeData))
        
        edgeData = add_fields(edgeData, ["inspection"], [object], [None])
        
        inspectionStationIDs = np.unique(edges["inspection"])
        if inspectionStationIDs[0] == b'':
            inspectionStationIDs = inspectionStationIDs[1:]
        
        self.stationIndexToStationID = inspectionStationIDs
        
        inspectionStationIndices = {iD:i for i, iD 
                                    in enumerate(inspectionStationIDs)}
        
        self.stationIDToStationIndex = inspectionStationIndices 
        edgesLen = len(edges)
        
        inspectionData = edgeData["inspection"]
        consideredIndices = np.nonzero(edges["inspection"][:,0])[0]
        inspectionData[consideredIndices] = [{inspectionStationIndices[iD]}
                                             for iD in edges["inspection"][
                                                        consideredIndices, 0]]
        consideredIndices = np.nonzero(edges["inspection"][:,1])[0]
        inspectionData[edgesLen:][consideredIndices] = [
                {inspectionStationIndices[iD]} for iD in 
                edges["inspection"][consideredIndices, 1]
                ]
        consideredIndices = np.nonzero(edges["inspection"][:,2])[0]
        for i, iD in zip(consideredIndices, 
                         edges["inspection"][consideredIndices, 2]):
            for j in i, i + edgesLen:
                if inspectionData[j]: 
                    inspectionData[j].add(inspectionStationIndices[iD])
                else:
                    inspectionData[j] = {inspectionStationIndices[iD]}
                
        self.prst("Creating graph")
        graph = FlexibleGraph(from_to, edgeData, vertexData["ID"], 
                              vertexData[["potentialViaVertex", "type"]],
                              replacementMode="shortest", lengthLabel="length")
        
        graph.add_vertex_attributes(("destinationID", "significant"), 
                                    (IDTYPE, bool),
                                    (b'', graph.vertices.array["type"] > 0))
        graph.set_default_vertex_data((0, True, 0, "", False))
        
        # adding virtual edges to represent destinations
        self.__add_virtual_destination_vertices(graph)
        if destinationToDestination:
            self.__add_virtual_destination_vertices(graph, 1)
        
        graph.remove_insignificant_dead_ends("significant")
        
        self.prst("Creating fast graph")
        super().__init__(graph, "length", "significant")
        
        self.vertices.array[:self.vertices.size]["significant"] = True
        
        self.sinkIndexToVertexIndex = self.vertices.get_array_indices("type", 2)
        self.sinkIndexToSinkID = self.vertices.array["ID"][
                                                self.sinkIndexToVertexIndex]
        order = np.argsort(self.sinkIndexToSinkID)
        self.sinkIndexToSinkID = self.sinkIndexToSinkID[order]
        self.sinkIndexToVertexIndex = self.sinkIndexToVertexIndex[order]
        
        self.sinkIDToSinkIndex = {iD:index for index, iD 
                                  in enumerate(self.sinkIndexToSinkID)}
        self.rawSinkIndexToVertexIndex = self.sinkIndexToVertexIndex
        
        self.sourceIndexToVertexIndex = self.vertices.get_array_indices("type", 1)
        
        self.sourceIndexToSourceID = self.vertices.array["ID"][
                                                self.sourceIndexToVertexIndex]
        order = np.argsort(self.sourceIndexToSourceID)
        self.sourceIndexToSourceID = self.sourceIndexToSourceID[order]
        self.sourceIndexToVertexIndex = self.sourceIndexToVertexIndex[order]
        self.rawSourceIndexToVertexIndex = self.sourceIndexToVertexIndex
        
        self.sourceIDToSourceIndex = {iD:index for index, iD 
                                      in enumerate(self.sourceIndexToSourceID)}
        self.sourcesConsidered = np.ones(self.sourceIndexToSourceID.size, 
                                         dtype=bool)
        
        
        self.postalCodeIndexToVertexIndex = self.vertices.get_array_indices("type", 3)
        
        self.prst("Transport network creation finished.")
        
    def preprocessing(self, preprocessingArgs=None):
        """Preprocesses the graph to make path search queries more efficient.
        
        This step is a necessary prerequisite to all path queries as implemented
        here.
        
        The 'reach' of each vertex is computed. The 'reach' is high if a vertex
        is at the center of a long shortest path (e.g. a highway).
        
        Parameters
        ----------
        preprocessingArgs : dict or iterable 
            Contains the arguments for 
            :py:class:`lopaths.graph.FlowPointGraph.preprocessing`. 
            If ``None``, reasonable default arguments will be chosen. 
            Refer to 
            :py:class:`lopaths.graph.FlowPointGraph.preprocessing`
            for a list of the possible arguments and their meaning
                
        """
        
        
        if preprocessingArgs is None:
            preprocessingArgs = self.preprocessingArgs
        
        if preprocessingArgs is None:
            FlowPointGraph.preprocessing(self, 1, 3, 3, 
                                         expansionBounds=(1.5, 1.5, 2.5),
                                         maxEdgeLength=30)
        else:
            if type(preprocessingArgs) == dict:
                FlowPointGraph.preprocessing(self, **preprocessingArgs)
            else:
                FlowPointGraph.preprocessing(self, *preprocessingArgs)
                
 
    def __add_virtual_destination_vertices(self, graph, vertexType=2):
        """        Creates vertices that represent the destinations
        
        """
        
        edges = graph.edges
        edgeArr = edges.get_array()
        consideredEdges = edgeArr[edgeArr["destinationID"] != np.array("", dtype=IDTYPE)]
        
        if vertexType == 1:
            destinationLabel = self._DESTINATION_SOURCE_LABEL
            destinationEdgeLabel = self._DESTINATION_SOURCE_EDGE_LABEL
        elif vertexType == 2:
            destinationLabel = self._DESTINATION_SINK_LABEL
            destinationEdgeLabel = self._DESTINATION_SINK_EDGE_LABEL
                                  
        consideredEdges = consideredEdges[np.argsort(consideredEdges["destinationID"])]
        if not consideredEdges.size:
            return
        
        self.prst("Adding virtual edges to represent destinations")
        lastdestinationID = -1
        
        newVertexLines = (graph.vertices.size-graph.vertices.space
                          + np.unique(consideredEdges["destinationID"]).size)
        if newVertexLines > 0:
            graph.vertices.expand(newVertexLines)
        
        newEdgeLines = (graph.edges.size-graph.edges.space
                          + consideredEdges.size)
        if newEdgeLines > 0:
            graph.edges.expand(newEdgeLines)
        
        for i, row in enumerate(consideredEdges):
            if lastdestinationID != row["destinationID"]:
                lastdestinationID = row["destinationID"]
                newVertexID = destinationLabel + row["destinationID"]
                graph.add_vertex(newVertexID, (newVertexID, True, vertexType,
                                               lastdestinationID, True))
            try:
                vertex = row["fromID"]
                if vertexType == 1:
                    fromTo = (newVertexID, vertex)
                else:
                    fromTo = (vertex, newVertexID)
                graph.add_edge(*fromTo, (destinationEdgeLabel + bytes(str(i), 'utf8'),
                                         row["length"], False, ''))
            except KeyError:
                edgeData = graph.get_edge_data(*fromTo, False)
                if row["length"] < edgeData["length"]:
                    edgeData["length"] = row["length"]
                    
    def find_shortest_distances(self):
        """Determines the shortest distances between all origins and destinations
        and from all postal code centres to the destinations
        
        """
        
        dists = self.find_shortest_distance_array(self.sourceIndexToVertexIndex, 
                                                  self.sinkIndexToVertexIndex)
        # treat disconnected sources and sinks well
        sourcesConsidered = np.min(dists, 1) < np.inf
        sinksConsidered = np.min(dists, 0) < np.inf
        if not (sourcesConsidered.all() and sinksConsidered.all()):
            dists = dists[sourcesConsidered][:,sinksConsidered]
            self.sinkIndexToVertexIndex = self.sinkIndexToVertexIndex[
                                                                sinksConsidered]
        
            self.sinkIndexToSinkID = self.vertices.array["ID"][
                                                    self.sinkIndexToVertexIndex]
        
            self.sinkIDToSinkIndex = {iD:index for index, iD 
                                      in enumerate(self.sinkIndexToSinkID)}
            self.update_sources_considered(considered=sourcesConsidered)
        
        if (dists == np.inf).any():
            warnings.warn("Some sources and sinks are disconnected though " +
                          "no source is separated from all sinks and " +
                          "vice versa. That is, the graph has separate " +
                          "subgraphs.")
            
        self.postalCodeDistances = self.find_shortest_distance_array(
                                            self.postalCodeIndexToVertexIndex, 
                                            self.sinkIndexToVertexIndex)
        
        if (self.postalCodeDistances == np.inf).any():
            warnings.warn("Some postal code areas and sinks are disconnected.")
            
        self.shortestDistances = dists
        
        if "sinksConsidered" in self.__dict__:
            tmp = np.nonzero(self.sinksConsidered)[0][~sinksConsidered]
            self.sinksConsidered[tmp] = False
        else:
            self.sinksConsidered = sinksConsidered
        self._pair_factor = np.array((self.shortestDistances.shape[1], 1))
    
    def update_sources_considered(self, 
            rawConsidered=None,
            considered=None):
        """Changes which origins are considered to fit the model. 
        
        
        If long-distance traffic is assumed to follow different mechanisms than 
        intermediate-distance traffic, it can be beneficial to fit two distinct
        models for traffic from different origins. 
        `update_sources_considered` determines which of the origins are 
        considered to fit the model. 
        
        Parameters
        ----------
        rawConsidered : bool[] 
            Boolean array determining which of the sources are considered.
            Must have the same size as the number of sources.
        considered : bool[]
            Boolean array determining which of the currently considered 
            sources remain considered. Must have the same size as the number 
            of currently considered sources
                
        """
        
        if "sourcesConsidered" not in self.__dict__:
            self.sourcesConsidered = np.ones(self.sourceIndexToSourceID.size, 
                                             dtype=bool)
        
        if rawConsidered is not None:
            self.sourcesConsidered = rawConsidered
        
        if considered is not None:
            tmp = np.nonzero(self.sourcesConsidered)[0][~considered]
            self.sourcesConsidered[tmp] = False
    
        self.sourceIndexToVertexIndex = self.rawSourceIndexToVertexIndex[
                                                        self.sourcesConsidered]
        self.sourceIndexToSourceID = self.vertices.array["ID"][
                                            self.sourceIndexToVertexIndex]
    
        self.sourceIDToSourceIndex = {iD:index for index, iD 
                                      in enumerate(
                                                self.sourceIndexToSourceID)}
        if "shortestDistances" in self.__dict__:
            del self.__dict__["shortestDistances"]
        
    def find_potential_routes(self, 
            stretchConstant=1.5, 
            localOptimalityConstant=.2, 
            acceptionFactor=0.667,
            rejectionFactor=1.333):
        """Searches potential routes of boaters. 
        
        For detailed documentation on the arguments, refer to 
        :py:class:`lopaths.graph.FlowPointGraph.find_alternative_paths`
        
        Parameters
        ----------
        stretchConstant : >=1
            Maximal length of the admissible paths in relation to shortest 
            path. Controls the length of the admissible paths. ``1`` refers
            to shortest paths only, `infinity` to all paths.
        localOptimalityConstant : [0, 1] 
            Fraction of the path that must be optimal. Controls how optimal 
            the admissible paths shall be. ``0`` refers to all paths, ``1`` 
            refers to shortest paths only.
        acceptionFactor : (0,1] 
            Relaxation factor for local optimality constraint. Approximation 
            factor to speed up the local optimality check. ``0`` accepts all 
            paths, ``1`` performs an exact check. Choose the
            largest feasible value. ``1`` is often possible.
        rejectionFactor : [1,2] 
            False rejection factor for local optimality constraint. 
            Approximation factor to speed up the local optimality
            check. ``1`` performs exact checks, ``2`` may reject paths that 
            are admissible but not locally optimal twice as much as required.
            Choose the smallest feasible value. ``1`` is often possible.
        
        """
        
        if not "shortestDistances" in self.__dict__:
            self.find_shortest_distances()
        
        return FlowPointGraph.find_alternative_paths(self, 
                                                self.sourceIndexToVertexIndex, 
                                                self.sinkIndexToVertexIndex,
                                                self.shortestDistances, 
                                                stretchConstant, 
                                                localOptimalityConstant, 
                                                acceptionFactor,
                                                rejectionFactor)
    
    def _pair_to_pair_index(self, pair):
        """Converts a tuple (OriginIndex, DesitnationIndex) to an integer index
        of the pair.
        
        """
        return np.sum(self._pair_factor*pair)


class BaseTrafficFactorModel(metaclass=DocMetaSuperclass):
    """Base class for traffic factor models.
    
    The traffic factor model is a gravity model yielding factors proportional
    to the vector flow between different origins and destinations. Admissible
    models should inherit this class and overwrite/implement its variables and
    methods.
    
    Objects of this class save the given covariate data as object variables 
    that can later be used to compute a factor proportional to the mean traffic 
    flow between sources and sinks.
    
    Parameters
    ----------
    sourceData : Struct[] 
        Array containing source covariates 
    sinkData : Struct[] 
        Array containing sink covariates
    postalCodeAreaData : Struct[] 
        Array containing population counts of postal code areas
    distances : double[] 
        Shortest distances between sources and sinks
    postalCodeDistances : double[] 
        Shortest distances between postal code areas and sinks
            
    """
    
    SIZE = 0
    "(`int`) -- Maximal number of parameters in the implemented model."
    
    # If only the name of the covariate is given, the data type will be assumed
    # to be double
    ORIGIN_COVARIATES = []
    """(`(str, type)[]`) -- The names and types of the covariates for the sources.
    If the type is not spcified, it will default to float.
    """
    
    DESTINATION_COVARIATES = []
    "(`(str, type=double)[]`) -- The names and types of the covariates for the sinks."
    
    PERMUTATIONS = None
    """(`bool[][]`) -- Parameter combinations to be considered when selecting the optimal model.
    
    The number of columns must match the maximal number of parameters of the 
    model (see :py:attr:`SIZE`)."""
            
    LABELS = np.array([], dtype="object")
    """(`str[]`) -- The names of the parameters in the implemented model.
    
    The length must match the maximal number of parameters of the 
    model (see :py:attr:`SIZE`)."""
    
    BOUNDS = np.array([])
    """(`(float, float)[]`) -- Reasonable bounds for the parameters (before conversion).
    
    The length must match the maximal number of parameters of the 
    model (see :py:attr:`SIZE`)."""
    
    def __init__(self, sourceData, sinkData, postalCodeAreaData, distances, 
                 postalCodeDistances):
        """Constructor"""
        pass
    
    @classmethod
    def _check_integrity(cls):
        """Checks whether derived classes are implementing the class correctly."""
        
        assert cls.SIZE == len(cls.LABELS)
        assert cls.SIZE == len(cls.BOUNDS)
        try: 
            cls.get_mean_factor(None, None, None)
        except NotImplementedError as e:
            raise e
        except Exception:
            pass
    
    def convert_parameters(self, dynamicParameters, considered):
        """Converts an array of given parameters to an array of standard (maximal)
        length and in the parameter domain of the model.
        
        Not all parameters may be considered in the model (to avoid overfitting)
        Furthermore, some parameters must be constrained to be positive or 
        within a certain interval. In this method, the parameter vector 
        (containing only the values of the free parameters) is transformed to 
        a vector in the parameter space of the model
        
        Parameters
        ----------
        dynamicParameters : double[] 
            Free parameters. The parameters that are not held constant.
        considered : bool[] 
            Which parameters are free? Is ``True`` at the entries corresponding 
            to the parameters that are free. ``considered`` must have exactly as 
            many ``True`` entries as the length of ``dynamicParameters``
        
        """
        result = np.full(len(considered), np.nan)
        result[considered] = dynamicParameters
        return result
    
    def get_mean_factor(self, params, considered, pair=None):
        """Returns a factor proportional to the mean traveller flow between the
        source-sink pair ``pair`` or all sources and sinks (if ``pair is None``)
        
        ~+~
        
        This method MUST be overwritten. Otherwise the model will through an 
        error.
        
        Parameters
        ----------
        params : double[] 
            Contains the free model parameters
        considered : bool[] 
            Which parameters are free? Is ``True`` at the entries corresponding 
            to the parameters that are free. ``considered`` must have exactly as 
            many ``True`` entries as the length of ``dynamicParameters``
        pair : (int, int) 
            Source-sink pair for which the factor shall be determined.
            This is the source-sink pair of interest (the indices of the source and
            the sink, NOT their IDs. If ``None``, the factors for all source-sink
            combinations are computed)
            
        """
        raise NotImplementedError()
    
    @add_doc(get_mean_factor)
    def get_mean_factor_autograd(self, params, considered):
        """Same as :py:meth:`get_mean_factor`, but must use autograd's functions 
        instead of numpy. 
        
        This function is necessary to compute derivatives with automatic 
        differentiation.
        
        ~+~
        
        To write this function, copy the content of :py:meth:`get_mean_factor` 
        and exchange ``np.[...]`` with ``ag.[...]``
        
        .. note:: autograd functions so not support in-place operations. 
            Therefore, an autograd-compatible implementation may be less efficient.
            If efficiency is not of greater concern, just use the autograd functions
            in `get_mean_factor` already and leave this method untouched.
            
        """
        return self.get_mean_factor(params, considered)
    
    @staticmethod
    def process_source_covariates(covariates):
        """Process source covariates before saving them.
        
        This method is applied to the source covariates before they are saved.
        The method can be used to compute derived covariates
        
        Parameters
        ----------
        covariates : float[]
            Covariates describing the repulsiveness of sources
        
        """
        return covariates
    
    @staticmethod
    def process_sink_covariates(covariates):
        """Process sink covariates before saving them.
        
        This method is applied to the sink covariates before they are saved.
        The method can be used to compute derived covariates
        
        Parameters
        ----------
        covariates : float[]
            Covariates describing the attractiveness of sinks
        
        """
        return covariates
    
    

class TrafficFactorModel(BaseTrafficFactorModel):
    """
    Gravity model for a factor proportional to the mean traffic flow between
    an origin and a destination.
    
    Can be overwritten to build a model with different covariates.
    All implemented methods and constants must be adjusted!
    """
    
    # The data in the covariate files must have the columns specified below in 
    # the corresponding order!
    ORIGIN_COVARIATES = [("population", 'double'),
                         ("anglers", 'double'),
                         ("canadian", bool)] 
    DESTINATION_COVARIATES = [("area", 'double'),
                       ("perimeter", 'double'),
                       ("campgrounds", 'int'),
                       ("pointsOfInterest", 'int'),
                       ("marinas", 'int'),
                       ("population", 'double')]
    
    def __init__(self, originData, destinationData, postalCodeAreaData, 
                 distances, postalCodeDistances):
        """
        Constructor
        
        See `BaseTrafficFactorModel.__init__`
        """
        super(TrafficFactorModel, self).__init__(originData, destinationData, 
                 postalCodeAreaData, distances, postalCodeDistances)
        
        self.population = originData["population"]
        self.anglers = originData["anglers"]
        self.canadian = originData["canadian"]
        self.lakeArea = destinationData["area"]
        self.lakePerimeter = destinationData["perimeter"]
        self.campgrounds = destinationData["campgrounds"]
        self.pointsOfInterest = destinationData["pointsOfInterest"]
        self.marinas = destinationData["marinas"]
        self.lakePopulation5km = destinationData["population"]
        self.postalCodePopulation = postalCodeAreaData["population"]
        self.postalCodeDists = postalCodeDistances * 0.001 
        self.dists = distances * 0.0001
        
    @staticmethod
    def process_sink_covariates(covariates):
        """#
        Convert number of campgrounds, pointsOfInterest, and marinas to 
        presence/absebce data to avoid identifiability issues
        """
        covariates["campgrounds"] = covariates["campgrounds"] > 0
        covariates["pointsOfInterest"] = covariates["pointsOfInterest"] > 0
        covariates["marinas"] = covariates["marinas"] > 0
        return covariates
    
    SIZE = 22
    
    LABELS = np.array([
        "base area",
        "area exponent",
        "base perimeter",
        "perimeter exponent",
        "campground factor",
        "campground exponent",
        "POI factor",
        "POI exponent",
        "marina factor",
        "marina exponent",
        "surrounding population factor",
        "base surrounding population",
        "surrounding population exponent",
        "base PCA distance",
        "PCA distance exponent",
        "PC population exponent",
        "base population",
        "population exponent",
        "base angler number",
        "angler number exponent",
        "canadian factor",
        "distance exponent",
        ])
    
    BOUNDS = np.array(((-2, 2000),
                       (0, 3),
                       (-2, 2000),
                       (0, 3),
                       (-5, 20),
                       (-7, 7),
                       (-5, 20),
                       (-7, 7),
                       (-5, 20),
                       (-7, 7),
                       (-5, 20),
                       (-7, 500),
                       (0, 3),
                       (-5, 7),
                       (-10, 0),
                       (0, 3),
                       (-7, 500),
                       (0, 5),
                       (-7, 500),
                       (0, 5),
                       (-3, 5),
                       (-5, 0)))
    
    #     A0     A^    Pr0     Pr^   
    PERMUTATIONS = [
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False,  False, True, False, True,  False, True, True],
         [True, False,  False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False,  False, True, False, True,  False, True, True],
         [True, False,  False, False,True,  False, True,  False,  True, False,  True, True,  False, False, False,  False, True, False, True,  False, True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False, False, True, False, False,  False, True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  False, False, False, False, True, False, False,  False, True, True],
         [False, False, False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False, False, True, False, True,  False, True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False,  False, False, False, True,  False, True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  False, False, False,  False, False, False, True,  False, True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False,  False, False, False, False,  False, True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False,  False, True, True, False,  False, False, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False,  False, False, True, False,  False, False, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False,  False, False, False, True,  False, True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  False, True, True, False, False,  False, False, False, True,  False, True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False,  False, False, False, False, False,True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  False, True, True, False, False,  False, False, False, False, False,True, True],
         [True, False, False, False, True,  False, False, False,  True, False,  True, True,  True, False, False,  False, False, False, True,  False, True, True],
         [True, False, False, False, False, False, False, False,  True, False,  True, True,  True, False, False,  False, False, False, True,  False, True, True],
         [True, False, False, False, False, False, False, False,  False,False, True, True,  True, False, False,  False, False, False, True,  False, True, True],
         [True, False, False, False, True,  False, False, False,  False,False, True, True,  True, False, False,  False, False, False, True,  False, True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  False,False, False, False, False,  False, False, False,True,  False, True, True],
    ]
    
    def convert_parameters(self,     
                           dynamicParameters:"(double[]) Free parameters", 
                           considered:"(bool[]) Which parameters are free"):
        """#
        Converts an array of given parameters to an array of standard (maximal)
        length and in the parameter domain of the model
        
        See `BaseTrafficFactorModel.convert_parameters`
        """
        
        result = [np.nan]*len(considered)
        j = 0
        if considered[0]:
            result[0] = convert_R_pos(dynamicParameters[j])
            j += 1
        
        if considered[1]:
            result[1] = dynamicParameters[j]
            j += 1
            
        for i in range(2, 13):    
            if considered[i]:
                result[i] = convert_R_pos(dynamicParameters[j])
                j += 1
        
        if considered[13]:
            result[13] = dynamicParameters[j]
            j += 1
        if considered[14]:
            result[14] = convert_R_pos(dynamicParameters[j])
            j += 1
        if considered[15]:
            result[15] = dynamicParameters[j]
            j += 1
        if considered[16]:
            result[16] = convert_R_pos(dynamicParameters[j])
            j += 1
        if considered[17]:
            result[17] = dynamicParameters[j]
            j += 1
        if considered[18]:
            result[18] = convert_R_pos(dynamicParameters[j])
            j += 1
        if considered[19]:
            result[19] = dynamicParameters[j]
            j += 1
        if considered[20]:
            result[20] = ag.exp(dynamicParameters[j])
            j += 1
        if considered[21]:
            result[21] = dynamicParameters[j]
            j += 1
        return result
    
    def get_mean_factor(self, params, considered, pair=None):
        """# 
        Gravity model for a factor proportional to the traffic flow
        between a jurisdiction and a lake.
        
        See `BaseTrafficFactorModel.get_mean_factor` for further details.
        """
        #return 1 #!!!!!!!!!!!!!!!!!!!!!!!!!!
        postalCodePopulation = self.postalCodePopulation
        if pair is None or tuple(pair)==(None, None):
            population = self.population
            anglers = self.anglers
            canadian = self.canadian
            lakeArea = self.lakeArea
            lakePerimeter = self.lakePerimeter
            campgrounds = self.campgrounds
            pointsOfInterest = self.pointsOfInterest
            marinas = self.marinas
            lakePopulation5km = self.lakePopulation5km
            dists = self.dists
        else:
            if type(pair) == np.ndarray:
                source = pair.T[0]
                sink = pair.T[1]
            else:
                transposed = list(zip(pair))
                source = np.array(transposed[0])
                sink = np.array(transposed[1])
            
            population = self.population[source].ravel()
            anglers = self.anglers[source].ravel()
            canadian = self.canadian[source].ravel()
            lakeArea = self.lakeArea[sink].ravel()
            lakePerimeter = self.lakePerimeter[sink].ravel()
            campgrounds = self.campgrounds[sink].ravel()
            pointsOfInterest = self.pointsOfInterest[sink].ravel()
            marinas = self.marinas[sink].ravel()
            lakePopulation5km = self.lakePopulation5km[sink].ravel()
            if sink is None:
                postalCodeDists = self.postalCodeDists
            else:
                postalCodeDists = self.postalCodeDists[:,sink]
            
            if sink is None:
                dists = self.dists[source]
            elif source is None:
                dists = self.dists[:, sink]
            else:
                dists = self.dists[source, sink]
        
        # lake size covariates
        cons_l0, cons_l1, cons_l2, cons_l3 = considered[:4]
        l0, l1, l2, l3 = params[:4]
        
        # lake infrastructure covariates
        li0, li1, li2, li3, li4, li5, li6, li7, li8, li9, li10, li11 = params[4:16]
        cons_li0, cons_li1, cons_li2, cons_li3, cons_li4, cons_li5, cons_li6, \
            cons_li7, cons_li8, cons_li9, cons_li10, cons_li11 = considered[4:16]
        
        # jurisdiction covariates
        j0, j1, j2, j3, j4 = params[16:21]
        cons_j0, cons_j1, cons_j2, cons_j3, cons_j4 = jur_cons = considered[16:21]
        
        # distance covariate
        d = params[21]
        cons_d = considered[21]
        
        exp = np.exp
        
        if cons_l1 or cons_l0:
            if cons_l0:
                lakeAttractivity = lakeArea / (l0 + lakeArea) 
                if cons_l1:
                    lakeAttractivity = np.power(lakeAttractivity, l1, 
                                                lakeAttractivity)
            else: 
                lakeAttractivity = np.power(lakeArea, l1)
        else:
            lakeAttractivity = np.ones(lakeArea.size)
        if cons_l3 or cons_l2:
            if cons_l2:
                lakeAttractivity2 = lakePerimeter / (l2 + lakePerimeter) 
                if cons_l3:
                    lakeAttractivity *= np.power(lakeAttractivity2, l3, 
                                                 lakeAttractivity2)
                else:
                    lakeAttractivity *= lakeAttractivity2
            else: 
                lakeAttractivity *= np.power(lakePerimeter, l3)
        
            
        lakeAttractivitySum = 1 
        if cons_li0: 
            if cons_li1:
                campgrounds = np.power(campgrounds, li1)
            lakeAttractivitySum = lakeAttractivitySum + li0*campgrounds
        if cons_li2:
            if cons_li3:
                pointsOfInterest = np.power(pointsOfInterest, li3)
            lakeAttractivitySum = lakeAttractivitySum + li2*pointsOfInterest
        if cons_li4:
            if cons_li5:
                marinas = np.power(marinas, li5)
            lakeAttractivitySum = lakeAttractivitySum + li4*marinas
        if cons_li6:
            if cons_li7:
                lakePopulation5km = lakePopulation5km / (lakePopulation5km + li7)
            if cons_li8:
                lakePopulation5km = np.power(lakePopulation5km, li8)
            lakeAttractivitySum = lakeAttractivitySum + li6*lakePopulation5km    
            
        lakeAttractivity *= lakeAttractivitySum    
            
        
        if cons_li11: 
            if not cons_li9:
                li9 = -1
            if cons_li10:
                postalCodeDists = postalCodeDists + li10
            
            tmp = np.sum(np.power(postalCodeDists, li9 
                                  ).__imul__(postalCodePopulation[:,None]), 0)
            tmp /= np.mean(tmp)
            lakeAttractivity *= np.power(tmp, li11)
        
        if not jur_cons.any():
            jurisdictionCapacity = np.ones(population.size)[:,None]
        else:
            if cons_j1 or cons_j0:
                if cons_j0:
                    population = population / (j0 + population)
                else:
                    population = population.copy()
                if cons_j1:
                    jurisdictionCapacity = np.power(population, j1, population)[:,None]
                else:
                    if type(population) == np.ndarray:
                        jurisdictionCapacity = population[:,None]
            else:
                jurisdictionCapacity = 1
                
            
            if cons_j3 or cons_j2:
                if cons_j2:
                    anglers = anglers / (j2 + anglers)
                else:
                    anglers = anglers.copy()
                if cons_j3:
                    jurisdictionCapacity *= np.power(anglers, j3, anglers)[:,None]
                else:
                    jurisdictionCapacity *= anglers[:,None]
            
            if cons_j4:
                jurisdictionCapacity *= exp(canadian*np.log(j4))[:,None]
            
        if cons_d:
            shortestDistances = np.power(dists, d)
        else: 
            shortestDistances = np.ones(dists.shape)
        
        return (lakeAttractivity * jurisdictionCapacity) * shortestDistances
    
    def get_mean_factor_autograd(self, params, considered):
        """#
        Autograd version of `get_mean_factor`. The numpy functions are replaced
        by autograd functions to allow for automatic differentiation. This is
        needed for efficient likelihood maximization.
        """
        #return 1 #!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        
        population = self.population
        anglers = self.anglers
        canadian = self.canadian
        lakeArea = self.lakeArea
        lakePerimeter = self.lakePerimeter
        campgrounds = self.campgrounds
        pointsOfInterest = self.pointsOfInterest
        marinas = self.marinas
        lakePopulation5km = self.lakePopulation5km
        postalCodeDists = self.postalCodeDists
        postalCodePopulation = self.postalCodePopulation
        
        exp = ag.exp
        power = ag.power
        
        # lake size covariates
        cons_l0, cons_l1, cons_l2, cons_l3 = considered[:4]
        l0, l1, l2, l3 = params[:4]
        
        # lake infrastructure covariates
        li0, li1, li2, li3, li4, li5, li6, li7, li8, li9, li10, li11 = params[4:16]
        cons_li0, cons_li1, cons_li2, cons_li3, cons_li4, cons_li5, cons_li6, \
            cons_li7, cons_li8, cons_li9, cons_li10, cons_li11 = considered[4:16]
        
        # jurisdiction covariates
        j0, j1, j2, j3, j4 = params[16:21]
        cons_j0, cons_j1, cons_j2, cons_j3, cons_j4 = jur_cons = considered[16:21]
        
        # distance covariate
        d = params[21]
        cons_d = considered[21]
        
        if cons_l1 or cons_l0:
            if cons_l0:
                lakeAttractivity = lakeArea / (l0 + lakeArea) 
                if cons_l1:
                    lakeAttractivity = power(lakeAttractivity, l1)
            else: 
                lakeAttractivity = power(lakeArea, l1)
        else:
            lakeAttractivity = ag.ones(lakeArea.size)
        if cons_l3 or cons_l2:
            if cons_l2:
                lakeAttractivity2 = lakePerimeter / (l2 + lakePerimeter) 
                if cons_l3:
                    lakeAttractivity *= power(lakeAttractivity2, l3)
                else:
                    lakeAttractivity *= lakeAttractivity2
            else: 
                lakeAttractivity *= power(lakePerimeter, l3)
                
        
        lakeAttractivitySum = 1 
        if cons_li0: 
            if cons_li1:
                campgrounds = power(campgrounds, li1)
            lakeAttractivitySum = lakeAttractivitySum + li0*campgrounds
        if cons_li2:
            if cons_li3:
                pointsOfInterest = power(pointsOfInterest, li3)
            lakeAttractivitySum = lakeAttractivitySum + li2*pointsOfInterest
        if cons_li4:
            if cons_li5:
                marinas = power(marinas, li5)
            lakeAttractivitySum = lakeAttractivitySum + li4*marinas
        if cons_li6:
            if cons_li7:
                lakePopulation5km = (lakePopulation5km+1e-200) / (lakePopulation5km + li7) #hack to prevent 0/0
            if cons_li8:
                lakePopulation5km = power(lakePopulation5km, li8)
            lakeAttractivitySum = lakeAttractivitySum + li6*lakePopulation5km    
            
        lakeAttractivity *= lakeAttractivitySum    
        
        if cons_li11: 
            if not cons_li9:
                li9 = -1
            if cons_li10:
                postalCodeDists = postalCodeDists + li10
            
            tmp = ag.sum(power(postalCodeDists, li9) 
                         * (postalCodePopulation[:,None]), 0)
            tmp = tmp / ag.sum(tmp) * tmp.size
            lakeAttractivity *= power(tmp, li11)
        
        if not jur_cons.any():
            jurisdictionCapacity = ag.ones(population.size)[:,None]
        else:
            if cons_j1 or cons_j0:
                if cons_j0:
                    population = population / (j0 + population)
                else:
                    population = population.copy()
                if cons_j1:
                    jurisdictionCapacity = ag.power(population, j1)[:,None]
                else:
                    jurisdictionCapacity = population[:,None]
            else:
                jurisdictionCapacity = 1
                
            
            if cons_j3 or cons_j2:
                if cons_j2:
                    anglers = anglers / (j2 + anglers)
                if cons_j3:
                    jurisdictionCapacity = (jurisdictionCapacity
                                            * power(anglers, j3)[:,None])
                else:
                    jurisdictionCapacity = (jurisdictionCapacity
                                            * anglers[:,None])
            
            if cons_j4:
                jurisdictionCapacity *= exp(canadian*ag.log(j4))[:,None]
        
        if cons_d:
            shortestDistances = power(self.dists, d)
        else: 
            shortestDistances = ag.ones(self.dists.shape)
        
        return (lakeAttractivity * jurisdictionCapacity) * shortestDistances
    

class HybridVectorModel(HierarchichalPrinter):
    """
    Class for the hybrid vector model.
    
    Brings the model compoents together and provides functionality to fit,
    analyze, and  apply the model.
    
    Parameters
    ----------
    fileName : str
        Name (without extension) of the file to which the model shall be saved.
    trafficFactorModel_class : class
        Class representing the gravity model; 
        must be inherited from :py:class:`BaseTrafficFactorModel`.
    destinationToDestination : bool
        If ``True``, the given origins will be ignored and
        routes will be sought from destinations to destinations. Note that destination to destination model 
        is not yet implemented to an extent that allows the fit of the gravity
        model.
    printerArgs : tuple 
        Arguments for the hierarchical printer. Can be ignored ingeneral.
    """
    
    def __init__(self, fileName, 
                 trafficFactorModel_class=None,
                 destinationToDestination=False, 
                 **printerArgs):
        """Constructor"""
        HierarchichalPrinter.__init__(self, **printerArgs)
        self.set_traffic_factor_model_class(trafficFactorModel_class)
        self.fileName = fileName
        self.destinationToDestination = destinationToDestination
        self.routeModel = None
    
    def save(self, fileName=None):
        """Saves the model to the file ``fileName``.vmm
        
        Parameters
        ----------
        fileName : str
            File name (without extension). If ``None``, the model's default 
            file name will be used.
        
        """
        if fileName is None:
            fileName = self.fileName
        if fileName is not None:
            self.prst("Saving the model as file", fileName+".vmm")
            if hasattr(self, "roadNetwork"):
                self.roadNetwork.lock = None
            saveobject.save_object(self, fileName+".vmm")
    
    @add_doc(TransportNetwork)
    def create_road_network(self, 
            fileNameEdges=None, 
            fileNameVertices=None,
            preprocessingArgs=None,
            edgeLengthRandomization=0.001
            ):
        """Creates and preprocesses a route network"""
        self.roadNetwork = TransportNetwork(fileNameEdges, 
                                                     fileNameVertices, 
                                                     self.destinationToDestination,
                                                     preprocessingArgs,
                                                     edgeLengthRandomization,
                                                     parentPrinter=self)  
        
        self.__check_origin_road_match()
        self.__check_destination_road_match()
        self.__check_postal_code_road_match()
        
        self.roadNetwork.preprocessing(preprocessingArgs)
        
        if "shortestDistances" in self.roadNetwork.__dict__:
            warnings.warn("The road network changes. The previous model" +
                          "and processing result are therefore inconsistent " +
                          "and will be removed.")
            self.routeModel = None
    
    def set_compliance_rate(self, complianceRate):
        """Sets the boaters' compliance rate (for stopping at inspection/survey 
        locations) 
        
        The rate is used for both fitting the model and optimizing inspection 
        station operation
        
        Parameters
        ----------
        complianceRate : float
            Proportion of agents stopping at survey/inspection stations.
            
        """
        self.complianceRate = complianceRate
        self.__erase_optimization_result()
    
    def read_postal_code_area_data(self, fileNamePostalCodeAreas):
        """Reads and saves data on postal code regions.
        
        Creates and saves an array with postal code area center vertex ID,  
        postal code, and population
        
        Parameters
        ----------
        fileNamePostalCodeAreas : str
            Name of a csv file with (ignored) header and columns separated by 
            ``,``. The following columns must be present in the specified order:
            
            ============ ================== =========
            Field        Type               Description
            ============ ================== =========
            Postal code  :py:data:`IDTYPE`  ID of the postal code area
            Vertex ID    :py:data:`IDTYPE`  ID of a vertex representing the postal code area (e.g.
                                            a vertex at the centre or population centre)
            population   int                population living in the postal code area. Can be the
                                            actual population count or the number in hundrets, thousands, etc.
                                            the results just have to be interpreted accordingly
            ============ ================== =========
            
        """
        self.prst("Reading postal code area data file", fileNamePostalCodeAreas)
        popData = np.genfromtxt(fileNamePostalCodeAreas, delimiter=",",  
                                skip_header = True, 
                                dtype = {"names":["postal_code", 
                                                  "vertexID", "population"], 
                                         'formats':[IDTYPE, IDTYPE, int]})
        popData.sort(order="vertexID")
        self.postalCodeAreaData = popData
        self.__check_postal_code_road_match()
        self.__erase_traffic_factor_model()
        self.__erase_optimization_result()
            
    def read_origin_data(self, fileNameOrigins):
        """Reads and saves data that can be used to determine the repulsiveness of origins in the vector traffic model.
        
        Parameters
        ----------
        fileNameOrigins : str
            Name of a csv file with (ignored) header and columns separated by 
            ``,``. The following columns must be present in the specified order
            
            ======================================================================= ============================== =========
            Field                                                                   Type                           Description
            ======================================================================= ============================== =========
            Origin ID                                                               :py:data:`IDTYPE`              ID of the origin. Must be coinciding with the 
                                                                                                                   respective ID used in the road network
            :py:attr:`ORIGIN_COVARIATES <BaseTrafficFactorModel.ORIGIN_COVARIATES>` ...                            Columns with the information and types specified
                                                                                                                   in the :py:class:`TrafficFactorModel` class. See                  
                                                                                                                   :py:attr:`BaseTrafficFactorModel.ORIGIN_COVARIATES`     
            ...
            ======================================================================= ============================== =========
            
        """
        self.prst("Reading origin data file", fileNameOrigins)
        dtype = [("originID", IDTYPE), ("infested", bool)]
        for nameType in self.__trafficFactorModel_class.ORIGIN_COVARIATES:
            if type(nameType) is not str and hasattr(nameType, "__iter__"): 
                if len(nameType) >= 2:
                    dtype.append(nameType[:2])
                else:
                    dtype.append((nameType[0], "double"))
            else:
                dtype.append((nameType, "double"))
        
        
        popData = np.genfromtxt(fileNameOrigins, delimiter=",", skip_header=True, 
                                dtype=dtype) 
        popData = self.__trafficFactorModel_class.process_source_covariates(popData)
        popData.sort(order="originID")
        #self.jurisdictionPopulationFactor = np.max(popData["population"])
        #popData["population"] /= self.jurisdictionPopulationFactor
        self.rawOriginData = popData
        self.__check_origin_road_match()
        self.__erase_optimization_result()
        self.__erase_traffic_factor_model()
        if ("roadNetwork" in self.__dict__):
            self.destinationData = popData[self.roadNetwork.sourcesConsidered]
    
    def read_destination_data(self, fileNameDestinations):
        """Reads and saves data that can be used to determine the attractiveness of destinations in the vector traffic model.
        
        Parameters
        ----------
        fileNameDestinations : str
            Name of a csv file with (ignored) header and columns separated by 
            ``,``. The following columns must be present in the specified order
            
            ================================================================================= ============================== =========
            Field                                                                             Type                           Description
            ================================================================================= ============================== =========
            Destination ID                                                                    :py:data:`IDTYPE`              ID of the destination. Must be coinciding with the 
                                                                                                                             respective ID used in the road network
            :py:attr:`DESTINATION_COVARIATES <BaseTrafficFactorModel.DESTINATION_COVARIATES>` ...                            Columns with the information and types specified
                                                                                                                             in the :py:class:`TrafficFactorModel` class. See                  
                                                                                                                             :py:attr:`BaseTrafficFactorModel.DESTINATION_COVARIATES`     
            ...
            ================================================================================= ============================== =========
        
        """
        self.prst("Reading destination data file", fileNameDestinations)
        
        dtype = [("destinationID", IDTYPE)]
        for nameType in self.__trafficFactorModel_class.DESTINATION_COVARIATES:
            if type(nameType) is not str and hasattr(nameType, "__iter__"): 
                if len(nameType) >= 2:
                    dtype.append(nameType[:2])
                else:
                    dtype.append((nameType[0], "double"))
            else:
                dtype.append((nameType, "double"))
                
        
        originData = np.genfromtxt(fileNameDestinations, delimiter=",", skip_header = True, 
                                 dtype = dtype)
        originData = self.__trafficFactorModel_class.process_sink_covariates(originData)
        originData.sort(order="destinationID")

        
        self.rawDestinationData = originData
        self.__check_destination_road_match()
        self.__erase_optimization_result()
        self.__erase_traffic_factor_model()
        
        if ("roadNetwork" in self.__dict__ 
                and "sinksConsidered" in self.roadNetwork.__dict__):
            self.originData = self.rawDestinationData[self.roadNetwork.sinksConsidered]
    
    
    def set_infested(self, originID, infested=True):
        """Chenges the infestation state of an origin with the given ID.
        
        Parameters
        ----------
        originID : :py:data:`IDTYPE` 
            ID of the origin whose state shall be changed
        infested : bool
            Infestation state. ``True`` means infested.
            
        """
        inds = np.nonzero(self.destinationData["originID"]==originID)[0]
        if not inds.size:
            raise ValueError("A jursidiction with ID {} does not exist".format(originID))
        self.destinationData["infested"][inds] = infested
        
    def __check_postal_code_road_match(self):
        """Checks whether the given vertex IDs in the postal code area data are 
        present in the road network
        
        """
        if ("roadNetwork" in self.__dict__ and 
                "postalCodeAreaData" in self.__dict__):
            popData = self.postalCodeAreaData
            if not (popData["vertexID"] 
                    == self.roadNetwork.vertices.get_array()["ID"][
                        self.roadNetwork.postalCodeIndexToVertexIndex]).all():
                raise ValueError("The vertexIDs of the postal code area centers"
                                 + " and the road network do not match.\n" 
                                 + "Maybe some postal code area data are "
                                 + "missing?")
    
    def __check_origin_road_match(self):
        """Checks whether the vertices given in the origin data are present 
        in the road network and vice versa
        
        """
        if self.destinationToDestination:
            return
        if ("roadNetwork" in self.__dict__ and 
            "rawOriginData" in self.__dict__):
            popData = self.rawOriginData
            if not (popData["originID"] 
                    == self.roadNetwork.vertices.get_array()["ID"][
                            self.roadNetwork.rawSourceIndexToVertexIndex]).all():
                raise ValueError("The originIDs of the population data and "
                                 + "the road network do not match.\n" 
                                 + "Maybe some population data are missing?")
    
    def __check_destination_road_match(self):
        """Checks whether all destinations for which we have covariates are present in the
        road network and vice versa
        
        """
        if ("roadNetwork" in self.__dict__ and "rawDestinationData" in self.__dict__):
            originData = self.rawDestinationData
            if (not originData.size 
                    == self.roadNetwork.rawSinkIndexToVertexIndex.size):
                raise ValueError("The numbers of destinations in the destination data " 
                                 + str(originData["destinationID"].size) 
                                 + " and the road network " 
                                 + str(self.roadNetwork.sinkIndexToSinkID.size) 
                                 + " do not match. Maybe some destination data are"
                                 + " missing?")
            L = self.roadNetwork._DESTINATION_SINK_LABEL
            for ID1, ID2 in zip(originData["destinationID"], 
                                self.roadNetwork.vertices.get_array(
                                    )["ID"][
                                    self.roadNetwork.rawSinkIndexToVertexIndex]):
                if not L + ID1 == ID2: # and False:
                    raise ValueError("The destinationIDs of the destination data and the road"
                                     + " network do not match.\n"
                                     + "Maybe some destination data are missing?")
    
    def find_shortest_distances(self):
        """Determines the shortest distances between all considered origins and 
        destinations, and destinations and postal code area centres
        
        See :py:meth:`TransportNetwork.find_shortest_distances`.
        
        """
        roadNetwork = self.roadNetwork
        
        reset = "shortestDistances" in roadNetwork.__dict__
        if reset:
            oldSinksConsidered = roadNetwork.sinksConsidered
            oldSourcesConsidered = roadNetwork.sourcesConsidered
        
        roadNetwork.find_shortest_distances()
        sourcesConsidered = roadNetwork.sourcesConsidered
        sinksConsidered = roadNetwork.sinksConsidered
        
        if reset:
            if not ((sourcesConsidered == oldSourcesConsidered).all()
                    and (sinksConsidered == oldSinksConsidered).all()):
                warnings.warn("The road network must have changed. " +
                              "Previous models are therefore inconsistent " +
                              "and will be removed.")
                self.routeModel = None
                del self.__dict__["countData"]
                del self.__dict__["pairCountData"]
            else:
                reset = False
        else:
            reset = True
        
        if reset:
            if "rawOriginData" in self.__dict__:
                self.destinationData = self.rawOriginData[
                                                        sourcesConsidered]
            if "rawDestinationData" in self.__dict__:
                self.originData = self.rawDestinationData[sinksConsidered]
                
    
    def set_origins_considered(self, considered=None, infested=None):
        """Determines which origins are considered in the model fit.
        
        It can happen that the model is to be fitted to model vectors coming 
        from specific origins, e.g. a model for long-distance traffic and
        a model for intermediate-distance traffic. In this case, ``considered`` 
        can be used to specify the origins considered to fit the model. 
        
        If ``infested`` is given and ``considered`` is NOT specified, then the model
        will be fitted to the origins with the given infestation status.
        E.g. ``infested=True`` will result in the model being fitted to estimate
        traffic from infested jursidictions only. All other data will be 
        ignored.
        
        Parameters
        ----------
        considered : bool[]
            Array determining which of the sources are considered
        infested : bool
            Select considered sources based on the infestation status
            
        """
        if considered is None:
            if infested is None:
                considered = np.ones(self.rawOriginData["infested"].size,
                                     dtype=bool)
            else:
                considered = self.rawOriginData["infested"]==infested
    
        if not (considered == self.roadNetwork.sourcesConsidered).all():
            self.roadNetwork.update_sources_considered(considered)
            self.routeModel = None
            for item in "countData", "pairCountData":
                try:
                    del self.__dict__[item]
                except KeyError:
                    pass
    
    @add_doc(TransportNetwork.find_potential_routes)
    def find_potential_routes(self, 
            stretchConstant=1.5, 
            localOptimalityConstant=.2, 
            acceptionFactor=0.667,
            rejectionFactor=1.333):
        """# Find potential vector routes."""
        
        result = self.roadNetwork.find_potential_routes(stretchConstant, 
                                                     localOptimalityConstant, 
                                                     acceptionFactor,
                                                     rejectionFactor)
        
        routeLengths, inspectedRoutes, stationCombinations = result
        routeParameters = (stretchConstant, localOptimalityConstant, 
                           acceptionFactor, rejectionFactor)
        routeModelData = {"routeLengths":routeLengths,
                          "inspectedRoutes":inspectedRoutes,
                          "stationCombinations":stationCombinations,
                          "routeParameters":routeParameters
                          }
        
        self.routeModel = routeModelData
            
    def __create_travel_time_distribution(self, index=None, longDist=True, 
                                          trafficData=None, parameters=None):
        """Create and fit the travel time model.
        
        .. todo:: Complete parameter documentation.
        
        Parameters
        ----------
        parameters : float[]
            If given, this will be the parameters of the travel time 
            distribution. If not given, the optimal parameters will be 
            determined via a maximum likelihood fit. 
            See :py:class:`traveltime_model.TrafficDensityVonMises`
            
        """
        if parameters is not None:
            return TrafficDensityVonMises(*parameters)
        
        travelTimeModel = TrafficDensityVonMises()
        if trafficData is None:
            if longDist:
                trafficData = self.longDistTimeData
            else:
                trafficData = self.restTimeData
        
        if index is None:
            timeData = np.concatenate([td.get_array() for td 
                                       in trafficData.values()])
        elif hasattr(index, "__iter__"):
            timeData = np.concatenate([trafficData[i].get_array() for i 
                                       in index])
        else:
            timeData = trafficData[index].get_array()
        
            
        travelTimeModel.maximize_likelihood(timeData["time"], 
                                            timeData["shiftStart"], 
                                            timeData["shiftEnd"],
                                            getCI=True)
        
        return travelTimeModel
                        
    @add_doc(__create_travel_time_distribution)
    def create_travel_time_distribution(self, parameters=None, fileName=None):
        """Create and fit the travel time model.
        
        Parameters
        ----------
        fileName : str
            If given, a plot with the density function of the distribution will
            be saved under the given name as pdf and png. 
            Do not include the file name extension.
            
        """
        self.prst("Fitting the travel time model")
        self.travelTimeModel = travelTimeModel = self.__create_travel_time_distribution(parameters=parameters)
        
        self.__erase_extrapolate_data()
        if fileName is not None:
            travelTimeModel.plot(None, False, fileName)
        self.prst("Temporal traffic distribution found.")
        
        
    def read_survey_data(self, fileNameObservations, pruneStartTime=11, 
                              pruneEndTime=16, properDataRate=None):
        """Reads the survey observation data.
        
        Parameters
        ----------
        fileNameObservations : str 
            Name of a csv file containing the road network. The file must be a 
            have a header (will be ignored) and the following columns, 
            separated by ``,``:
            
            ============ ============================== =========
            Field        Type                           Description
            ============ ============================== =========
            stationID    :py:data:`IDTYPE`              ID of the survey location
            dayID        :py:data:`IDTYPE`              ID for the day of the survey (e.g. the date)
            shiftStart   [0, 24), `optional`            Start time of the survey shift
            shiftEnd     [0, 24), `optional`            End time of the survey shift
            time         [0, 24), `optional`            Time when the agent was observed
            fromID       :py:data:`IDTYPE`, `optional`  ID of the origin of the agent 
            toID         :py:data:`IDTYPE`, `optional`  ID of the destination of the agent
            relevant     bool                           Whether or not this agent is a potential vector
            ============ ============================== =========
            
            The times must be given in the 24h format. For example, 2:30PM 
            translates to ``14.5``.
            
            Missing or inconsistent data will either be ignored 
            (if ``relevant==False``) or incorporated as 'unknown' 
            (if ``relevant==True``). All applicable data will be used to fit
            the temporal traffic distribution.
        pruneStartTime : float
            Some parts of the extended model analysis require that only data 
            collected within the same time frame are considered. 
            ``pruneStartTime`` gives the start time of this time frame. 
            It should be chosen so that many survey shifts include the entire 
            time interval [``pruneStartTime``, ``pruneEndTime``].
        pruneEndTime : float
            End of the unified time frame (see :py:obj:`pruneStartTime`).
        properDataRate : float
            Fraction of agents providing inconsistent, incomplete, or wrong
            data. I not given, the rate will be estimated from the data.
            
        """
        
        
        self.prst("Reading boater data", fileNameObservations)
        self.__erase_extrapolate_data()
        self.__erase_optimization_result()
        self.__erase_travel_time_model()
        self.__erase_route_choice_model()
        
        dtype = {"names":["stationID", "dayID", "shiftStart", "shiftEnd", 
                          "time", "fromID", "toID", "relevant"], 
                 'formats':[IDTYPE, IDTYPE, 'double', 'double', 'double', 
                            IDTYPE, IDTYPE, bool]}
        
        surveyData = np.genfromtxt(fileNameObservations, delimiter=",",  
                                   skip_header = True, 
                                   dtype = dtype)
        surveyData = np.append(surveyData, np.zeros(1, dtype=surveyData.dtype))
        
        self.prst("Determining observed daily boater counts")
        
        pairCountData = np.zeros((self.roadNetwork.sourceIndexToVertexIndex.size, 
                                  self.roadNetwork.sinkIndexToVertexIndex.size), 
                                  dtype=int)
        
        shiftDType = [("dayIndex", int),
                      ("stationIndex", int),
                      ("shiftStart", 'double'),
                      ("shiftEnd", 'double'),
                      ("countData", object),
                      ("prunedCountData", object),
                      ("totalCount", object),
                      ("prunedCount", object),
                      ]
        
        
        dayDType = [("shifts", object),
                    ("countData", object),
                    ]
        
        dayData = FlexibleArray(10000, dtype=dayDType)
        shiftData = FlexibleArray(10000, dtype=shiftDType)
        
        timeDType = {"names":["shiftStart", "shiftEnd", "time"], 
                      'formats':['double', 'double', 'double']}
        longDistTimeData = defaultdict(lambda: FlexibleArray(1000, dtype=timeDType))
        restTimeData = defaultdict(lambda: FlexibleArray(1000, dtype=timeDType))
        dayIDToDayIndex = dict()
        
        considered = surveyData["stationID"] != b''
        surveyData = surveyData[considered]
        
        dayArr = surveyData["dayID"]
        stationArr = surveyData["stationID"]
        shiftStartArr = surveyData["shiftStart"]
        shiftEndArr = surveyData["shiftEnd"]
        fromArr = surveyData["fromID"]
        toArr = surveyData["toID"]
        timeArr = surveyData["time"]
        relevantArr = surveyData["relevant"]
        sourceIndices = self.roadNetwork.sourceIDToSourceIndex
        sinkIndices = self.roadNetwork.sinkIDToSinkIndex
        
        a = surveyData[["stationID", "dayID", "shiftStart", "shiftEnd"]]
        b = np.roll(a, -1)
        newShift = ~((a == b) | ((a != a) & (b != b))) 
        newShift[-1] = True
        shiftStartIndex = 0
        l = self.roadNetwork._DESTINATION_SOURCE_LABEL if self.destinationToDestination else b''
        L = self.roadNetwork._DESTINATION_SINK_LABEL
        
        rejectedCount = 0
        acceptedCount = 0
        for shiftEndIndex in np.nonzero(newShift)[0]+1:
            obsdict = defaultdict(lambda: 0)
            try:
                stationIndex = self.roadNetwork.stationIDToStationIndex[
                                                    stationArr[shiftStartIndex]]
            except KeyError:
                shiftStartIndex = shiftEndIndex
                continue
            
            if dayArr[shiftStartIndex] == b'':
                shiftStartIndex = shiftEndIndex
                continue
            
            shiftStart = shiftStartArr[shiftStartIndex]
            shiftEnd = shiftEndArr[shiftStartIndex]
            if shiftStart >= shiftEnd:
                shiftStart -= 24
                
                # we assume that we have no shifts longer than 12 hours
                # therefore, too long over night shifts are considered an error.
                if shiftEnd - shiftStart >= 12:
                    shiftStart = shiftEnd = np.nan
            
            validObservationCount = np.isfinite(timeArr[shiftStartIndex:shiftEndIndex]).sum()
            if np.isnan(shiftStart):
                if not validObservationCount:
                    shiftStartIndex = shiftEndIndex
                    continue
                t = timeArr[shiftStartIndex:shiftEndIndex]
                if np.isfinite(shiftEnd):
                    if shiftEnd-12 < 0 and (t >= (shiftEnd-12)%24).any():
                        shiftStart = np.min(t[t >= (shiftEnd-12)%24]) - 24
                    else:
                        shiftStart = np.nanmin(t)
                else:
                    t = np.sort(t[np.isfinite(t)])
                    diffs = (t - np.roll(t, 1)) % 24
                    switch = np.argmax(diffs)
                    shiftStart = t[switch]
                    shiftEnd = t[switch-1]
                    if switch:
                        shiftStart -= 24
            elif np.isnan(shiftEnd):
                if not validObservationCount:
                    shiftStartIndex = shiftEndIndex
                    continue
                t = timeArr[shiftStartIndex:shiftEndIndex]
                if shiftStart+12 > 24 and (t <= (shiftStart+12)%24).any():
                    shiftEnd = np.max(t[t <= (shiftStart+12)%24])
                    shiftStart -= 24
                else:
                    shiftEnd = np.nanmax(t)
    
            if not shiftStart < shiftEnd:
                shiftStartIndex = shiftEndIndex
                continue
             
            if shiftStart < 0:
                timeArr[shiftStartIndex:shiftEndIndex][
                    timeArr[shiftStartIndex:shiftEndIndex] > shiftEnd] -= 24
            
            # Here we assume that pruneStartTime and pruneEndTime are >= 0
            if ((shiftStart <= pruneStartTime and shiftEnd >= pruneEndTime) or
                    (shiftStart < 0 and shiftStart+24 <= pruneStartTime)):
                prunedCount = 0
                includePrunedShift = True
                prunedObsdict = defaultdict(lambda: 0)
            else: 
                includePrunedShift = False
                prunedObsdict = None
            
            totalCount = 0
            for i in range(shiftStartIndex, shiftEndIndex):
                time = timeArr[i]
                
                if time < 0:
                    includeTime = False
                    includePrunedShift = False
                elif not np.isnan(time):
                    if not shiftStart < time < shiftEnd:
                        #shiftStartIndex = shiftEndIndex
                        continue
                    includeTime = True
                else: 
                    includeTime = False
                
                # put totalCount here, if observations that were not included
                # in the optimization shall be counted as well
                #totalCount += 1
                
                try:
                    if not relevantArr[i]:
                        raise KeyError()
                    
                    fromIndex = sourceIndices[l+fromArr[i]]
                    toIndex = sinkIndices[L+toArr[i]]
                    
                    # comment, if not infested jursidictions shall be included
                    #if not self.destinationData[fromIndex]["infested"]:
                    #if self.destinationData[fromIndex]["infested"]:
                    #    raise KeyError()
                    
                    obsdict[(fromIndex, toIndex)] += 1
                    pairCountData[fromIndex, toIndex] += 1
                    totalCount += 1
                    if includePrunedShift:
                        if pruneStartTime <= time <= pruneEndTime:
                            prunedCount += 1
                            prunedObsdict[(fromIndex, toIndex)] += 1
                    if includeTime:
                        longDistTimeData[stationIndex].add_tuple((shiftStart, 
                                                                  shiftEnd, 
                                                                  time))
                except KeyError:
                    if includeTime:
                        restTimeData[stationIndex].add_tuple((shiftStart, 
                                                              shiftEnd, time))
                    rejectedCount += relevantArr[i]
                
            if not includePrunedShift:
                prunedCount = -1
            
            dayID = dayArr[shiftStartIndex]
            if dayID not in dayIDToDayIndex:
                dayIndex = dayData.add_tuple(([], obsdict))
                dayIDToDayIndex[dayID] = dayIndex
            else: 
                dayIndex = dayIDToDayIndex[dayID]
                dayObsdict = dayData[dayIndex]["countData"]
                for pair, count in obsdict.items():
                    dayObsdict[pair] += count
            
            shiftIndex = shiftData.add_tuple((dayIndex, stationIndex, 
                                              shiftStart, shiftEnd, 
                                              dict(obsdict), 
                                              (dict(prunedObsdict) if 
                                                 prunedObsdict is not None 
                                                 else None), 
                                 totalCount, prunedCount))
            
            acceptedCount += totalCount
            dayData[dayIndex]["shifts"].append(shiftIndex) 
            shiftStartIndex = shiftEndIndex
        
        self.pruneStartTime = pruneStartTime
        self.pruneEndTime = pruneEndTime
        self.longDistTimeData = dict(longDistTimeData)
        self.restTimeData = dict(restTimeData)
        self.shiftData = shiftData.get_array()
        self.dayData = dayData.get_array()
        _properDataRate = acceptedCount/(rejectedCount+acceptedCount)
        self.prst("Fraction of complete data:", _properDataRate)
        if properDataRate is None:
            self.properDataRate = _properDataRate
        else:
            self.properDataRate = properDataRate
            self.prst("Set properDataRate to given value", properDataRate)
        
        dayCountData = self.dayData["countData"]
        for i, d in enumerate(dayCountData):
            dayCountData[i] = dict(d)
        
        self.pairCountData = pairCountData
        
        self.prst("Daily boater counts and time data determined.")
    
    
    def __erase_optimization_result(self):
        """Resets the gravity model to an unfitted state."""
        if self.routeModel is None:
            return
        routeModelData = self.routeModel
        if routeModelData is None:
            return
        try:
            del routeModelData["AIC"]
            del routeModelData["parameters"] 
            del routeModelData["covariates"]
        except KeyError:
            pass
    
    def __erase_traffic_factor_model(self):
        """Erases the gravity model."""
        self.__dict__.pop("trafficFactorModel", None)
    
    def __erase_travel_time_model(self):
        """Erases the travel time model."""
        self.__erase_route_choice_model()
        try:
            del self.__dict__["travelTimeModel"]
        except Exception:
            pass
    
    def __erase_route_choice_model(self):
        """Erases the route choice model."""
        routeModelData = self.__dict__.get("routeModel", None)
        if routeModelData:
            routeModelData.pop("routeChoiceModel", None)
            
    def __erase_extrapolate_data(self):    
        """Erases the prepared observation data for the model fit."""
        warn = False
        routeModelData = self.routeModel
        if routeModelData is None:
            return
        if "fullCountData" in routeModelData:
            warn = True
            self.routeModel = {
                "routeLengths":routeModelData["routeLengths"],
                "inspectedRoutes":routeModelData["inspectedRoutes"],
                "stationCombinations":routeModelData["stationCombinations"]
                } 
        if warn:
            warnings.warn("The boater data or traffic distribution changed. "
                          + "Therefore, the boater extrapolation and previously "
                          + "found models become inconsistent and are deleted.")
        
    
    def create_route_choice_model(self, redo=False):
        """Creates and fits the route choice model.
        
        Parameters
        ----------
        redo : bool
            Whether the route choice model shall be refitted if it has been 
            fitted already. If set to ``True``, the previous fit will be ignored.
        
        """
    
        self.prst("Creating route choice model")
        
        try:
            routeModelData = self.routeModel
            if not redo and "routeChoiceModel" in routeModelData:
                self.prst("Route model has already been created.")
                return False
        except Exception:
            warnings.warn("Route candidates must be computed before a route "
                          + "model can be created. Nothing has been done. Call"
                          + "model.find_potential_routes(...)")
            return False
            
        self.increase_print_level()
        shiftData = self.shiftData
        
        self.prst("Preparing road model")
        
        routeChoiceModel = RouteChoiceModel(parentPrinter=self)
        routeChoiceModel.set_fitting_data(self.dayData, shiftData, 
                                          routeModelData["inspectedRoutes"], 
                                          routeModelData["routeLengths"], 
                                          self.travelTimeModel,
                                          self.complianceRate, self.properDataRate)
        routeModelData["routeChoiceModel"] = routeChoiceModel
            
        self.decrease_print_level()
        return True    
            
        
    def preprocess_count_data(self, redo=False):
        """Takes the raw survey data and preprocesses them for the model fit.
        
        Parameters
        ----------
        redo : bool
            Whether the task shall be repeated if it had been done before.
            If set to ``True``, the earlier result be ignored.
        
        """
        
        self.prst("Extrapolating the boater count data")
        try:
            routeModelData = self.routeModel
            if not redo and "fullCountData" in routeModelData:
                self.prst("Count data have already been extrapolated.")
                return False
        except Exception:
            warnings.warn("Route candidates must be computed before a route "
                          + "model can be created. Nothing has been done. Call"
                          + "model.find_potential_routes(...)")
            return False
        
        self.increase_print_level()
        self.prst("Extrapolating the count data")
        sourceIndexToSourceID = self.roadNetwork.sourceIndexToSourceID
        sinkIndexToSinkID = self.roadNetwork.sinkIndexToSinkID
        stationIndexToStationID = self.roadNetwork.stationIndexToStationID
        
        if not redo and "fullCountData" in routeModelData:
            self.prst("Boater count data exist already")
            return False
        self.increase_print_level()
        
        self.prst("Extrapolating boater count data")
        
        countDType = {"names":["pairIndex", "p_shift", "count"], 
                      'formats':[int, 'double', int]}
        fullCountData = FlexibleArray(10000, dtype=countDType)
        
        shiftDType = {"names":["p_shift", "usedStationIndex", "shiftStart",
                               "shiftEnd"], 
                      'formats':['double', int, float, float]}
        shiftData = np.empty(self.shiftData.size, dtype=shiftDType)
        
        noiseDType = {"names":["pairIndex", "p_shift", "count"], 
                      'formats':[int, 'double', int]}
        observedNoiseData = FlexibleArray(10000, dtype=noiseDType)
        
        
        inspectedRoutes = routeModelData["inspectedRoutes"]
        routeLengths = routeModelData["routeLengths"]
        countData = self.shiftData
        factor = np.array((self.roadNetwork.shortestDistances.shape[1], 1))
        
        usedStationIndexToStationIndex = np.unique(countData["stationIndex"]
                                                   )
        self.usedStationIndexToStationIndex = usedStationIndexToStationIndex
        
        stationIndexToUsedStationIndex = {index:i for i, index in 
                                  enumerate(usedStationIndexToStationIndex)}
            
        stationDType = {"names":["pairIndices", "consideredPathLengths"], 
                        'formats':[object, object]}
        stationData = np.empty(usedStationIndexToStationIndex.size, 
                               dtype=stationDType)
        
        for i, stationIndex in enumerate(usedStationIndexToStationIndex):
            if stationIndex in inspectedRoutes:
                consideredPathLengths = [
                    [routeLengths[pair[0], pair[1], pathIndex] for pathIndex 
                    in pathIndices] for pair, pathIndices 
                    in inspectedRoutes[stationIndex].items()]
            else:
                consideredPathLengths = []
                
            if stationIndex in inspectedRoutes:
                pairIndices = np.array([np.sum(factor*pair) for pair in 
                                        inspectedRoutes[stationIndex].keys()],
                                       dtype=int)
            else:
                pairIndices = np.array([], dtype=int)
            stationData[i] = (pairIndices, 
                              list_to_csr_matrix(consideredPathLengths))
        
        consideredPathLengths = []
        travelTimeModel = self.travelTimeModel.interval_probability
        allObservations = 0
        falseObservations = FlexibleArray(1000, 
                                          dtype=[("stationID", IDTYPE),
                                                 ("fromID", IDTYPE),
                                                 ("toID", IDTYPE)])
        counter = Counter(self.shiftData.size, 0.01)
        
        for i, row in enumerate(countData):
            p_shift = travelTimeModel(row["shiftStart"], row["shiftEnd"])
            stationIndex = row["stationIndex"]
            obsdict = row["countData"]
            percentage = counter.next()
            if percentage: self.prst(percentage, percent=True)
            
            inspectedRoutesDict = inspectedRoutes.get(stationIndex, [])
            fullCountData.extend([(np.sum(factor*pair), p_shift, 
                                   count) for pair, count in 
                                  obsdict.items() 
                                  if pair in inspectedRoutesDict]
                                 )
            
            consideredPathLengths.extend(
                [routeLengths[pair[0], pair[1], pathIndex] for pathIndex 
                 in inspectedRoutesDict[pair]] for pair in obsdict.keys()
                                         if pair in inspectedRoutesDict
                                         )
            
            observedNoise = (set(obsdict) 
                             - set(inspectedRoutes.get(stationIndex, set()))
                             )
            observedNoiseData.extend([(np.sum(factor*pair), p_shift, 
                                       obsdict[pair]) 
                                      for pair in observedNoise])
            
            # number of all possible pairs - number of covered pairs
            #  + number of false observations                          
            shiftData[i] = (p_shift, 
                            stationIndexToUsedStationIndex[stationIndex],
                            row["shiftStart"], row["shiftEnd"])
            allObservations += len(obsdict)
            
            stationID = stationIndexToStationID[stationIndex]
            for pair in observedNoise:
                falseObservations.add_tuple(
                    (stationID, sourceIndexToSourceID[pair[0]],
                     sinkIndexToSinkID[pair[1]]))
            
        consideredPathLengths = list_to_csr_matrix(consideredPathLengths)
        
        self.prst(falseObservations.size, "out of", allObservations, 
                  "boaters were observed at an unexpected location.",
                  "({:6.2%})".format(falseObservations.size/
                                     allObservations))
        
        falseObservations = falseObservations.get_array()
        
        print(falseObservations)
        df = pd.DataFrame(falseObservations)
        df.to_csv(self.fileName + "FalseObservations.csv", index=False)
        
        routeModelData["fullCountData"] = fullCountData.get_array()
        routeModelData["consideredPathLengths"] = consideredPathLengths
        routeModelData["observedNoiseData"] = observedNoiseData.get_array()
        routeModelData["shiftData"] = shiftData
        routeModelData["stationData"] = stationData
        self.decrease_print_level()
        self.decrease_print_level()
        return True
    
    def _get_k_value(self, params, considered, pair=None):
        return HybridVectorModel._get_k_value_static(
            params, considered, self.trafficFactorModel, pair)
    
    @staticmethod
    def _get_k_value_static(params, considered, trafficFactorModel, pair=None):
        q = params[1]   
        c0 = params[0] * (1-q) / q  # reparameterization k->mu
        return trafficFactorModel.get_mean_factor(params[2:], considered[2:], 
                                                  pair) * c0
    
    @staticmethod
    def _get_k_value_autograd_static(params, considered, trafficFactorModel):
        q = params[1]   
        c0 = params[0] * (1-q) / q  # reparameterization k->mu
        return trafficFactorModel.get_mean_factor_autograd(params[2:], 
                                                           considered[2:]) * c0
    
    @staticmethod
    def _convert_parameters_static(params, considered, trafficFactorModel):
        
        return ([convert_R_pos(params[0]), convert_R_0_1(params[1])] 
                 + trafficFactorModel.convert_parameters(params[2:], 
                                                         considered[2:]))
    
    def _convert_parameters(self, params, considered):
        
        return HybridVectorModel._convert_parameters_static(
            params, considered, self.trafficFactorModel)
        
    
    @staticmethod
    def _negLogLikelihood(parameters, routeChoiceParams, considered, pairIndices, 
                          stationPairIndices, observedNoisePairs,
                          routeProbabilities, 
                          stationRouteProbabilities,
                          stationIndices,
                          p_shift, shiftDataP_shift, observedNoiseP_shift, 
                          p_shift_mean, stationKs, 
                          kSumNotObserved,
                          approximationNumber,
                          counts, observedNoiseCounts, trafficFactorModel): 
        """
        parameters:
        A:    lake area or perimeter
        P:    population
        F:    Number of Anglers (Fisherman)
        D:    shortest distance
        D_r:  length of the respective path
        PC:   population of postal code area
        D_PC: travel time from postal code area to lake
        N_c:  Number of campgrounds
        N_m:  Number ofmarinas
        N_poi:  Number of points of interest (toilets, tourist infos, viewpoints, parks, attractions, picnic sites)
        CA:   1, if province is in Canada, else 0
        k (prop. to mean): c0 * (A / (d0 + A))^d1 * (1 + d2*N_c^d3 + d4*N_poi^d5 + d6*N_m^d7 + d8*(N_pop/(d9+N_pop))^d10) 
                              * (sum((D_PC+d11)^d12)*PC / <mean)^d13
                              * (P / (P + d14))^d15 * (F / (F + d16))^d17 * e^(CA+d18) * D^d19
        q (prop. to 1-mean/variance): c1         #* D^d4
        Edge choice probability: (1-c2) * (sum(D_r^c3)/normalization) + c2*c4
        Probability to be off-route and observable: c2*c4
        """
        
        params = HybridVectorModel._convert_parameters_static(
                                    parameters, considered, trafficFactorModel)
        
        #print("staticParameters", staticParameters)
        #print("restparams", restparams)
        
        c1 = params[1]
        c2, _, c4 = routeChoiceParams
        
        kMatrix = HybridVectorModel._get_k_value_static(
                    params, considered, trafficFactorModel)
        
        #print("np.isfinite(kMatrix).all()", np.isfinite(kMatrix).all())
        
        kMatrix = kMatrix.ravel() 
        k = kMatrix[pairIndices]
        
        """
        normConstants = sparsepowersum(routeLengths, c3)
        routeProbabilities = sparsepowersum(consideredPathLengths, c3)
            
        routeProbabilities = (routeProbabilities 
                              / normConstants[pairIndices]  
                              * (1-c2) + c2 * c4)
        
        """
        
        #print("np.isfinite(routeProbabilities).all()", np.isfinite(routeProbabilities).all())
        
        aq = routeProbabilities * p_shift * c1
        qqm = (1-c1)/(1-c1+aq) 
        
        # Liekelihood associated with observations > 0 on expected routes
        likelihoodOnWays = np.sum(nbinom.logpmf(counts, k, qqm), 0)
        
        for i, stPairIndices, sRP in zip(itercount(),
                                         stationPairIndices, 
                                         stationRouteProbabilities):
            """
            
            stationRouteProbabilities = 
            
            x1 = sparsepowersum(pathLengths, c3)
            x2 = normConstants[stPairIndices]
            stationRouteProbabilities = np.add(np.multiply(np.divide(x1, x2, x2), (1-c2), x2),
                        c2 * c4, x2)
            """
            qr = sRP * c1
            ks = kMatrix[stPairIndices]
            x4 = np.log(1-c1)-np.log((1-c1) + p_shift_mean * qr)
            x4 *= ks
            stationKs[i, 0] = np.sum(x4, 0)
            
            tmpFact = -qr / ((1-c1) + qr*p_shift_mean)
            tmp = tmpFact.copy()
            stationKs[i, 1] = np.sum(tmp * ks, 0)
            for j in range(2, approximationNumber+1):
                tmp *= tmpFact
                stationKs[i, j] = np.sum(tmp * ks, 0)
            kSumNotObserved[i] = np.sum(ks, 0)
        
        shiftP_shiftAppr = shiftDataP_shift - p_shift_mean
        p_shiftAppr = p_shift - p_shift_mean
        
        # Likelihood associated with obervations = 0 on expected routes
        # (here: assume all observations have been 0)
        likelihoodNotObservedOnWays = np.sum(stationKs[stationIndices, 0], 0)
        
        # Compute the share of likelihoodNotObservedOnWays that is wrong, 
        # because we actually observed something
        qr = c1 * routeProbabilities
        likelihoodNotObservedOnWaysNeg = (np.log(1-c1)
                                          - np.log((1-c1)+(qr*p_shift_mean))
                                          )
        
        #print("np.isfinite(likelihoodNotObservedOnWaysNeg).all()", np.isfinite(likelihoodNotObservedOnWaysNeg).all())
        
        for j in range(1, approximationNumber+1):
            likelihoodNotObservedOnWays += np.sum(
                np.power(shiftP_shiftAppr, j)*stationKs[stationIndices, j] 
                / j, 0
                )
            likelihoodNotObservedOnWaysNeg += np.power(-qr*p_shiftAppr
                                                       / ((1-c1)
                                                          +qr*p_shift_mean), 
                                                       j) / j
        likelihoodNotObservedOnWaysNeg = np.sum(
                                        k*likelihoodNotObservedOnWaysNeg, 0)
        likelihoodNotObservedOnWays -= likelihoodNotObservedOnWaysNeg
        
        
        # Likelihood associated with observations = 0 on not expected
        # routes (assume first that all pairs have no route via this spot)
        aq = (c2 * c4 * c1) * shiftDataP_shift
        qq2ml = np.log(1-c1) - np.log(1-c1+aq)
        likelihoodRestWays = (np.sum(qq2ml) * np.sum(kMatrix, 0) 
                              - np.sum(qq2ml*kSumNotObserved[stationIndices],
                                       0))
        
        
        aq = (c2 * c4 * c1) * observedNoiseP_shift
        qq3m = (1-c1)/(1-c1+aq)  
        k2 = kMatrix[observedNoisePairs]
        likelihoodFalseWays = np.sum(nbinom.logpmf(observedNoiseCounts,
                                       k2, qq3m), 0)
        likelihoodRestWays -= np.sum(k2 * np.log(qq3m), 0)
        
        
        #print("np.isfinite(likelihoodOnWays).all()", np.isfinite(likelihoodOnWays).all())
        #print("np.isfinite(likelihoodRestWays).all()", np.isfinite(likelihoodRestWays).all())
        #print("np.isfinite(likelihoodFalseWays).all()", np.isfinite(likelihoodFalseWays).all())
        #print("np.isfinite(likelihoodNotObservedOnWays).all()", np.isfinite(likelihoodNotObservedOnWays).all())
        
        result = (- likelihoodOnWays - likelihoodRestWays 
                  - likelihoodFalseWays - likelihoodNotObservedOnWays)
        
        if np.any(np.isnan(result)): 
            return np.inf 
        
        '''
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        dayCounts = np.array([2, 3, 0, 1, 0, 0, 0])
        
        result -= np.sum(nbinom.logpmf(dayCounts,
                                       kMatrix[:dayCounts.size], 1-c1), 0)*10
        
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        '''
        return result
    
    @staticmethod
    def _negLogLikelihood_autograd(parameters, routeChoiceParams,
                                   considered, pairIndices, 
                                   stationPairIndices, observedNoisePairs,
                                   routeProbabilities, 
                                   stationRouteProbabilities,
                                   stationIndices,
                                   p_shift, shiftDataP_shift, 
                                   observedNoiseP_shift, 
                                   p_shift_mean, stationNumber, 
                                   approximationNumber,
                                   counts, observedNoiseCounts, 
                                   trafficFactorModel, 
                                   convertParameters=True): 
        
        log = ag.log
        
        if convertParameters:
            params = HybridVectorModel._convert_parameters_static(
                                parameters, considered, trafficFactorModel)
        else:
            params = parameters
        
        c1 = params[1]
        c2, _, c4 = routeChoiceParams
            
        kMatrix = HybridVectorModel._get_k_value_autograd_static(
                    params, considered, trafficFactorModel)
        
        kMatrix = kMatrix.reshape((kMatrix.size,)) 
        
        k = kMatrix[pairIndices]
        
        aq = routeProbabilities * p_shift * c1
        qqm = (1-c1)/(1-c1+aq) # = 1-aq/(1-c1+aq)
        
        stationKs = ag.zeros((stationNumber, approximationNumber+1), 
                             dtype=object)
        if isinstance(c1, ag.float):
            kSumNotObserved = ag.zeros(stationNumber)
        else:
            kSumNotObserved = ag.zeros(stationNumber, dtype="object")
            #kSumNotObserved = AutogradStorage(stationNumber) #ag.zeros(stationNumber, dtype="object")
        
        # Liekelihood associated with observations > 0 on expected routes
        likelihoodOnWays = ag.sum(nbinom_logpmf(counts, k, qqm), 0)
        for i, stPairIndices, sRP in zip(itercount(),
                                                 stationPairIndices, 
                                                 stationRouteProbabilities):
            qr = sRP * c1
            
            ks = kMatrix[stPairIndices]
            x4 = log(1-c1)-log((1-c1) + p_shift_mean * qr)
            x4 *= ks
            stationKs[i, 0] = ag.sum(x4, 0)
            tmpFact = -qr / ((1-c1) + qr*p_shift_mean)
            tmp = tmpFact + 0
            stationKs[i, 1] = ag.sum(tmp * ks, 0)
            for j in range(2, approximationNumber+1):
                tmp *= tmpFact
                stationKs[i, j] = ag.sum(tmp * ks, 0)
            kSumNotObserved[i] = ag.sum(ks, 0)
        
        shiftP_shiftAppr = shiftDataP_shift - p_shift_mean
        p_shiftAppr = p_shift - p_shift_mean
        
        # Likelihood associated with obervations = 0 on expected routes
        # (here: assume all observations have been 0)
        likelihoodNotObservedOnWays = ag.sum(stationKs[stationIndices, 0], 0)
        
        # Compute the share of likelihoodNotObservedOnWays that is wrong, 
        # because we actually observed something
        qr = c1 * routeProbabilities
        likelihoodNotObservedOnWaysNeg = (log(1-c1)
                                          - log((1-c1)+(qr*p_shift_mean))
                                          )
        
        likelihoodNotObservedOnWays += ag.sum(shiftP_shiftAppr
                                              *stationKs[stationIndices, 1] , 0)
        
        likelihoodNotObservedOnWaysNegTmp = -qr*p_shiftAppr / ((1-c1)
                                                               + qr*p_shift_mean)
        likelihoodNotObservedOnWaysNeg += likelihoodNotObservedOnWaysNegTmp
        for j in range(2, approximationNumber+1):
            shiftP_shiftAppr *= shiftP_shiftAppr
            likelihoodNotObservedOnWays += ag.sum(
                shiftP_shiftAppr*stationKs[stationIndices, j] 
                / j, 0
                )
            likelihoodNotObservedOnWaysNegTmp *= likelihoodNotObservedOnWaysNegTmp
            likelihoodNotObservedOnWaysNeg += likelihoodNotObservedOnWaysNegTmp / j
        likelihoodNotObservedOnWaysNeg = ag.sum(k*likelihoodNotObservedOnWaysNeg, 0)
        likelihoodNotObservedOnWays -= likelihoodNotObservedOnWaysNeg
        
        
        # Likelihood associated with observations = 0 on not expected
        # routes (assume first that all pairs have no route via this spot)
        aq = (c2 * c4 * c1) * shiftDataP_shift
        qq2ml = log(1-c1) - log(1-c1+aq)
        
        prd = 0
        for I, K in enumerate(stationIndices):
            prd = prd + qq2ml[I] * kSumNotObserved[K]
            #prd = prd + kSumNotObserved[K]
            
        likelihoodRestWays = (ag.sum(qq2ml) * ag.sum(kMatrix) # * shiftNumber 
                              - prd)
        """
        likelihoodRestWays = (ag.sum(qq2ml) * ag.sum(kMatrix, 0) # * shiftNumber 
                              - ag.sum(qq2ml*kSumNotObserved[stationIndices],
                                       0))
        if type(kSumNotObserved) == AutogradStorage:
            kSumNotObserved = kSumNotObserved.data #to_autograd_array(kSumNotObserved)
        likelihoodRestWays = (ag.sum(qq2ml) * ag.sum(kMatrix, 0) # * shiftNumber 
                              #- ag.sum(kSumNotObserved*qq2ml[0]))
                              - sum(qq2ml*kSumNotObserved[stationIndices]))
        """
        
        aq = (c2 * c4 * c1) * observedNoiseP_shift
        qq3m = (1-c1)/(1-c1+aq)  
        k2 = kMatrix[observedNoisePairs]
        likelihoodFalseWays = ag.sum(nbinom_logpmf(observedNoiseCounts,
                                       k2, qq3m), 0)
        likelihoodRestWays -= ag.sum(k2 * log(qq3m), 0)
        
        result = (- likelihoodOnWays - likelihoodRestWays 
                  - likelihoodFalseWays - likelihoodNotObservedOnWays)
        #print("result", result)
        if isinstance(result, ag.float) and ag.isnan(result): 
            return ag.inf 
        
        '''
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        dayCounts = np.array([2, 3, 0, 1, 0, 0, 0])
        
        result -= ag.sum(nbinom_logpmf(dayCounts,
                                       kMatrix[:dayCounts.size], 1-c1), 0)*10
        
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        '''
        
        
        return result


    @staticmethod
    def _get_nLL_funs(extrapolateCountData, trafficFactorModel,
                      routeChoiceParams, complianceRate, properDataRate,
                      considered, approximationNumber=3):
        if considered is None:
            considered = np.ones(20, dtype=bool)
        
        fullCountData = extrapolateCountData["fullCountData"]
        consideredPathLengths = extrapolateCountData["consideredPathLengths"] 
        observedNoiseData = extrapolateCountData["observedNoiseData"] 
        shiftDataP_shift = extrapolateCountData["shiftData"]["p_shift"] * complianceRate * properDataRate
        stationIndices = extrapolateCountData["shiftData"]["usedStationIndex"]
        stationPathLengths = extrapolateCountData["stationData"][
                                                        "consideredPathLengths"]
        stationPairIndices = extrapolateCountData["stationData"]["pairIndices"]
        
        p_shift = fullCountData["p_shift"] * complianceRate * properDataRate
        pairIndices = fullCountData["pairIndex"]
        counts = fullCountData["count"]
        routeLengths = extrapolateCountData["routeLengths"].data
        
        observedNoiseP_shift = observedNoiseData["p_shift"] * complianceRate * properDataRate
        observedNoiseCounts = observedNoiseData["count"]
        observedNoisePairs = observedNoiseData["pairIndex"]
        p_shift_mean = np.mean(shiftDataP_shift)
        stationNumber = stationPairIndices.size
        stationKs = np.zeros((stationNumber, approximationNumber+1))
        kSumNotObserved = np.zeros(stationNumber)
        
        _negLogLikelihood = HybridVectorModel._negLogLikelihood
        _negLogLikelihood_autograd = HybridVectorModel._negLogLikelihood_autograd
        
        
        c2, c3, c4 = routeChoiceParams
        
        normConstants = sparsepowersum(routeLengths, c3)
        routeProbabilities = sparsepowersum(consideredPathLengths, c3)
        routeProbabilities = (routeProbabilities 
                              / normConstants[pairIndices]  
                              * (1-c2) + c2 * c4)
        
        stationRouteProbabilities = []
        for stPairIndices, pathLengths in zip(stationPairIndices, 
                                              stationPathLengths):
            x1 = sparsepowersum(pathLengths, c3)
            x2 = normConstants[stPairIndices]
            stationRouteProbabilities.append(
                np.add(np.multiply(np.divide(x1, x2, x2), 
                                   (1-c2), x2), c2 * c4, x2)
                )
        
        
        def negLogLikelihood(params): 
            return _negLogLikelihood(params, routeChoiceParams, considered, 
                   pairIndices, stationPairIndices, 
                   observedNoisePairs, routeProbabilities, 
                   stationRouteProbabilities, stationIndices, p_shift, 
                   shiftDataP_shift, observedNoiseP_shift, p_shift_mean, 
                   stationKs, kSumNotObserved, 
                   approximationNumber, counts, observedNoiseCounts, 
                   trafficFactorModel)
        
        def negLogLikelihood_autograd(params, convertParameters=True):
            return _negLogLikelihood_autograd(params, routeChoiceParams,
                   considered, pairIndices, stationPairIndices, 
                   observedNoisePairs, routeProbabilities, 
                   stationRouteProbabilities, stationIndices, p_shift, 
                   shiftDataP_shift, observedNoiseP_shift, p_shift_mean, 
                   stationNumber, approximationNumber, counts, 
                   observedNoiseCounts, trafficFactorModel,
                   convertParameters=convertParameters)
        
        jac = grad(negLogLikelihood_autograd)
        hess = hessian(negLogLikelihood_autograd)

        return negLogLikelihood, negLogLikelihood_autograd, jac, hess, None
    
    def maximize_log_likelihood(self, considered=None, approximationNumber=3,
                                flowParameters=None, x0=None):
        routeChoiceParameters = self.routeModel["routeChoiceModel"].parameters
        
        return HybridVectorModel.maximize_log_likelihood_static(
                self.routeModel, self.trafficFactorModel, routeChoiceParameters,
                self.complianceRate, self.properDataRate, considered, 
                approximationNumber, flowParameters, x0)
    
    @staticmethod
    def maximize_log_likelihood_static(extrapolateCountData,
                                       trafficFactorModel,
                                       routeChoiceParams,
                                       complianceRate,
                                       properDataRate,
                                       considered, 
                                       approximationNumber=3,
                                       flowParameters=None,
                                       x0=None):
                 
        negLogLikelihood, negLogLikelihood_autograd, jac, hess, hessp = \
                HybridVectorModel._get_nLL_funs(extrapolateCountData, 
                                                      trafficFactorModel, 
                                                      routeChoiceParams,
                                                      complianceRate,
                                                      properDataRate,
                                                      considered, 
                                                      approximationNumber)
        
        
        if flowParameters is None:
            
            """
            print("Maximal difference to p_shift approximation point:",
                  np.max(np.abs(p_shift_mean-shiftDataP_shift)))
            print("Mean difference to p_shift approximation point:",
                  np.mean(np.abs(p_shift_mean-shiftDataP_shift)))
            """
    
            bounds = [(-15, 10), (-10, 0.5)]
            
            considered[:2] = True
            
            for bound in trafficFactorModel.BOUNDS[considered[2:]]:
                bounds.append(tuple(bound))
                
            if x0 is None:
                np.random.seed()
                
                x0 = np.ones(np.sum(considered))
                negLogLikelihood(x0)
                negLogLikelihood_autograd(x0)
                
                result = op.differential_evolution(negLogLikelihood, bounds, 
                                                   popsize=20, maxiter=20, #300, 
                                                   #popsize=20, maxiter=2, 
                                                   disp=True)
                print(considered)          
                print("GA result", result)
                x0 = result.x.copy()
                result.xOriginal = HybridVectorModel._convert_parameters_static(
                    result.x, considered, trafficFactorModel)
                result.jacOriginal = jac(result.xOriginal, False)
            else:
                result = op.OptimizeResult(x=x0, 
                                           success=True, status=0,
                                           fun=negLogLikelihood(x0), 
                                           nfev=1, njev=0,
                                           nhev=0, nit=0,
                                           message="parameters checked")
                result.xOriginal = HybridVectorModel._convert_parameters_static(
                    result.x, considered, trafficFactorModel)
                result.jacOriginal = jac(result.xOriginal, False)
            
            
            """
            class RandomDisplacement(op._basinhopping.RandomDisplacement):
                def __call__(self, x):
                    x += self.random_state.uniform(-self.stepsize, self.stepsize, np.shape(x))
                    x[1] = max(x[1], bounds[1][0]+1e-3)
                    x[2] = max(x[2], bounds[2][0]+1e-3)
                    return x
            
            for i, pair in enumerate(bounds):
                bounds[i] = (pair[0]*2 if pair[0] < 0 else pair[0], 2*pair[1])
            
            result = op.basinhopping(negLogLikelihood, x0, 1, # stepsize=1,
                                     #take_step=RandomDisplacement(1), 
                                     #minimizer_kwargs={"method":"L-BFGS-B",
                                     #minimizer_kwargs={"method":"Nelder-Mead",
                                     #minimizer_kwargs={"method":"Powell",
                                     minimizer_kwargs={"method":"SLSQP",
                                                       "options":{"maxiter":1200,
                                                                  "iprint":2,
                                                                  #"ftol":1e-8
                                                                  "bounds":bounds,
                                                                  #"disp":True
                                                                  }},
                                                                  #'ftol': 1e-06}},
                                     interval=5)
            """
            
            print(negLogLikelihood(x0))
            print(negLogLikelihood_autograd(x0))
            
            result2 = op.minimize(negLogLikelihood_autograd, x0, method="L-BFGS-B",
                                  jac=jac, hess=hess,
                                  bounds=None, options={"maxiter":800,
                                                        "iprint":2})
            print(considered)          
            result2.xOriginal = HybridVectorModel._convert_parameters_static(
                    result2.x, considered, trafficFactorModel)
            result2.jacOriginal = jac(result2.xOriginal, False)
            print("L-BFGS-B", result2)
            
            #if result2.fun < result.fun:
            x0 = result2.x.copy()
            result = result2
            
            result2 = op.minimize(negLogLikelihood_autograd, x0, jac=jac, 
                                  hess=hess, bounds=None, 
                                  options={"maxiter":800, 
                                           "iprint":2},
                                  method="SLSQP")
            print(considered)          
            result2.xOriginal = HybridVectorModel._convert_parameters_static(
                    result2.x, considered, trafficFactorModel)
            result2.jacOriginal = jac(result2.xOriginal, False)
            print("SLSQP", result2)         
            if result2.fun < result.fun:
                x0 = result2.x.copy()
                result = result2
            
            def testHess(x):
                h = hess(x)
                if not np.isfinite(h).all():
                    print("Hessian is infinite")
                    print("considered:", considered)
                    print("x:", x)
                    print("f:", negLogLikelihood_autograd(x))                    
                    print("jac:", jac(x))
                    print("hess:", h)
                return h
            """
            try:
                result2 = op.minimize(negLogLikelihood_autograd, result.x, jac=jac, 
                                      hessp=hessp, bounds=None, 
                                      method="CG",
                                      options={"maxiter":300, "disp":True})
            except ValueError:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          file=sys.stdout)
                result2 = op.OptimizeResult(x=x0, 
                                            success=False, status=1,
                                            fun=np.inf, 
                                            message="ValueError thrown")
            print(considered)          
            statP, dynP = HybridVectorModel._convert_parameters_static(result2.x,
                                                                      considered, trafficFactorModel)
            result2.xOriginal = statP + dynP
            result2.jacOriginal = jac(result2.xOriginal, False)
            print("CG", result2)         
            if result2.fun < result.fun:
                result = result2
            """
            try:
                result2 = op.minimize(negLogLikelihood_autograd, result.x, jac=jac, 
                                      hess=hess, bounds=None, 
                                      method="trust-exact",
                                      options={"maxiter":500, 
                                               "disp":True})
            except ValueError:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          file=sys.stdout)
                result2 = op.OptimizeResult(x=x0, 
                                            success=False, status=1,
                                            fun=np.inf, 
                                            message="ValueError thrown")
                
            if np.sum(considered) == 12:
                #line_profile = line_profiler.LineProfiler(negLogLikelihood)
                #line_profile.runcall(negLogLikelihood, x0)
                #line_profile.print_stats()
                #profile("negLogLikelihood(x0)", globals(), locals())
                pass
            print(considered)          
            result2.xOriginal = HybridVectorModel._convert_parameters_static(
                    result2.x, considered, trafficFactorModel)
            result2.jacOriginal = jac(result2.xOriginal, False)
            print("trust-exact", result2)         
            if result2.fun < result.fun:
                result = result2
            
        else:
            result = op.OptimizeResult(x=flowParameters, 
                                       success=True, status=0,
                                       fun=negLogLikelihood(flowParameters), 
                                       jac=jac(flowParameters), 
                                       nfev=1, njev=1,
                                       nhev=1, nit=0,
                                       message="parameters checked")
            result.xOriginal = HybridVectorModel._convert_parameters_static(
                flowParameters, considered, trafficFactorModel)
            result.jacOriginal = jac(result.xOriginal, False)
            
            checkParametersO = HybridVectorModel._convert_parameters_static(
                flowParameters, considered, trafficFactorModel)
            #profile("negLogLikelihood(flowParameters)", globals(), locals())
            #profile("negLogLikelihood_autograd(flowParameters)", globals(), locals())
            #profile("negLogLikelihood_autograd(checkParametersO, False)", globals(), locals())
            #profile("jac(flowParameters)", globals(), locals())
            #profile("hess(flowParameters)", globals(), locals())
            """
            j = Jacobian(negLogLikelihood_autograd)
            h = Hessian(negLogLikelihood_autograd)
            h2 = Jacobian(jac)
            profile("j(flowParameters)", globals(), locals())
            profile("h(flowParameters)", globals(), locals())
            profile("h2(flowParameters)", globals(), locals())
            profile("hessp(flowParameters)", globals(), locals())
            
            J = j(flowParameters)
            JAC = jac(flowParameters)
            print("np.max(np.abs(J-JAC))", np.max(np.abs(J-JAC)))
            H = h(flowParameters)
            HESS = hess(flowParameters)
            print("np.max(np.abs(H-HESS))", np.max(np.abs(H-HESS)))
            H2 = h2(flowParameters)
            print("np.max(np.abs(H2-HESS))", np.max(np.abs(H2-HESS)))
            """
            
            #profile("jac(checkParametersO, False)", globals(), locals())
            #line_profile.runcall(hess, flowParameters)
            #line_profile.print_stats()
            #profile("hess(checkParametersO, False)", globals(), locals())
            #benchmarking
            def testFun(parms, parmIndex):
                cons = np.ones_like(parms, dtype=bool)
                cons[parmIndex] = False
                h = hess(parms)
                hh = h[np.ix_(cons, cons)]
                result = np.dot(np.linalg.inv(hh), h[parmIndex][cons])
                print(result)
                return result
            
            try:
                #profile("testFun(flowParameters, 0)", globals(), locals())
                #profile("testFun(flowParameters, 3)", globals(), locals())
                #profile("testFun(flowParameters, 6)", globals(), locals())
                pass
            except np.linalg.LinAlgError:
                print("LinalgError")
        
        try:
            
            raise Exception()
            """
            step = 0.05
            #hess2 = Hessian(negLogLikelihood, step, method='central')
            
            if np.isnan(result.x).any():
                result.hess = np.nan
            else:
                result.hess = hess(result.x) #hess2(result.x)
                
                H = hess(result.xOriginal, False) #hess2(result.x)
                
                considered2 = np.concatenate((np.ones(5,dtype=bool),considered))
                H = H[np.ix_(considered2, considered2)]
                
                result.hessOriginal = H
                
                def detTest(m, i):
                    v = np.ones(m.shape[0], dtype=bool)
                    v[i] = False
                    return np.abs(np.linalg.det(m[v][:,v]))
                hDim = H.shape[0]
                m1 = m2 = m3 = m4 = 0
                for i in range(hDim): 
                    mm = detTest(H, i)
                    if mm > m1:
                        m1 = mm
                        maxi = i
                    for j in range(i+1, hDim):
                        mm = detTest(H, [i,j])
                        if mm > m2:
                            m2 = mm
                            maxij = [i,j]
                        for k in range(j+1, hDim):
                            mm = detTest(H, [i,j, k])
                            if mm > m3:
                                m3 = mm
                                maxijk = [i,j,k]
                            for l in range(k+1, hDim):
                                mm = detTest(H, [i,j, k, l])
                                if mm > m4:
                                    m4 = mm
                                    maxijkl = [i,j,k,l]
                try:
                    result.maxi = maxi
                except Exception:
                    result.maxi = np.nan
                try:
                    result.maxij = maxij
                except Exception:
                    result.maxij = np.nan
                try:
                    result.maxijk = maxijk
                except Exception:
                    result.maxijk = np.nan
                try:
                    result.maxijkl = maxijkl
                except Exception:
                    result.maxijkl = np.nan
                """
        except Exception:
            warnings.warn("Could not compute Hessian.")
            result.hess = np.nan
            result.hessOriginal = np.nan
        return result
    
    
    def investigate_profile_likelihood(self, x0, extrapolateCountData, 
                                       trafficFactorModel, routeChoiceParams,
                                       complianceRate,
                                       properDataRate,
                                       considered, 
                                       approximationNumber=3,
                                       **optim_args):
        
        negLogLikelihood, negLogLikelihood_autograd, jac, hess, hessp = \
                HybridVectorModel._get_nLL_funs(extrapolateCountData, 
                                                      trafficFactorModel, 
                                                      routeChoiceParams,
                                                      complianceRate,
                                                      properDataRate,
                                                      considered, 
                                                      approximationNumber) 
        
        self.prst("Investigating the profile likelihood")
        
        self.increase_print_level()
        
        if not "fun0" in optim_args:
            self.prst("Determining logLikelihood")
            optim_args["fun0"] = -negLogLikelihood_autograd(x0)
        if not "hess0" in optim_args:
            self.prst("Determining Hessian of logLikelihood")
            optim_args["hess0"] = -hess(x0)
        
        dim = len(x0)
        
        result = np.zeros((dim, 2))
        
        labels = ["c0", "q"] + list(self.trafficFactorModel.LABELS[considered[2:]])
        
        indices, directions = zip(*iterproduct(range(dim), (-1, 1)))
        
        inspectedRoutes = extrapolateCountData["inspectedRoutes"]
        del extrapolateCountData["inspectedRoutes"]
        
        const_args = [extrapolateCountData, trafficFactorModel, 
                      routeChoiceParams, self.complianceRate, self.properDataRate,
                      considered]
        
        self.prst("Creating confidence intervals")
        #try: #, max_workers=13
        with ProcessPoolExecutor(const_args=const_args) as pool:
            mapObj = pool.map(HybridVectorModel._find_profile_CI_static, 
                              indices, repeat(x0), directions, 
                              repeat(approximationNumber), repeat(optim_args))
            
            
            for index, direction, r in zip(indices, directions, mapObj):
                result[index][(0 if direction==-1 else 1)
                              ] = np.array(self._convert_parameters(r.x, 
                                        considered))[considered][index]
        
        self.prst("Printing confidence intervals and creating profile plots")
        self.increase_print_level()
        
        x0Orig = np.array(self._convert_parameters(x0, considered))[considered]
        
        for index, intv in enumerate(result):
            start, end = intv
            strs = []
            for v in start, x0Orig[index], end:
                if (0.01 < np.abs(v) < 1e6) or v == 0:
                    strs.append("{:10.4f}".format(v))
                else:
                    strs.append("{:<10}".format(str(v)))
            self.prst("CI for {:<40}: [{} --- {} --- {}]".format(
                labels[index], *strs))
            
        self.decrease_print_level()
        
        extrapolateCountData["inspectedRoutes"] = inspectedRoutes
        self.decrease_print_level()
    
    @staticmethod
    def _find_profile_CI_static(extrapolateCountData, 
                                trafficFactorModel, routeChoiceParams,
                                complianceRate,
                                properDataRate,
                                considered, 
                                index, x0, direction,
                                approximationNumber=3, 
                                profile_LL_args={}):
                 
        
        negLogLikelihood, negLogLikelihood_autograd, jac, hess, hessp = \
                HybridVectorModel._get_nLL_funs(extrapolateCountData, 
                                                      trafficFactorModel, 
                                                      routeChoiceParams,
                                                      complianceRate,
                                                      properDataRate,
                                                      considered, 
                                                      approximationNumber) 
        
        negLogLikelihood_autograd_ = lambda x: -negLogLikelihood_autograd(x)   
        jac_ = lambda x: -jac(x)   
        hess_ = lambda x: -hess(x)   
        
        return find_profile_CI_bound(index, direction, x0, negLogLikelihood_autograd_, jac_, hess_, 
                                     **profile_LL_args)
    
    
    def fit_route_model(self, refit=False, guess=None, 
                        improveGuess=False, disp=True, get_CI=True):
        
        self.increase_print_level()
        
        routeModelData = self.routeModel
        routeModel = routeModelData["routeChoiceModel"]
        if not refit and routeModel.fitted:
            self.prst("A route choice model does already exist. I skip",
                      "this step. Enforce fitting with the argument",
                      "refit=True")
            return False
        if not routeModel.prepared:
            self.prst("The route choice model has not been prepared for model",
                      "fit. I ignore it.",
                      "Call create_route_choice_models if you want to",
                      "use the model.")
            return False
        
        self.prst("Fitting route choice model")
        
        self.increase_print_level()
        routeModel.fit(guess, improveGuess, disp)
        self.decrease_print_level()
        self.prst("Constructing confidence intervals for route",
                  "choice model")
        
        if not (guess is not None and improveGuess==False) and get_CI:
            self.increase_print_level()
            fileName = self.fileName
            if not os.access(fileName, os.F_OK): os.makedirs(fileName)
            fileName = os.path.join(fileName, fileName)
            routeModel.get_confidence_intervals(fileName)
            self.decrease_print_level()
        self.decrease_print_level()
        
        return True
            
    def set_traffic_factor_model_class(self, trafficFactorModel_class=None):
        if trafficFactorModel_class is not None:
            if self.__dict__.pop("__trafficFactorModel_class", None) == trafficFactorModel_class:
                return
            trafficFactorModel_class._check_integrity()
            self.__trafficFactorModel_class = trafficFactorModel_class
        self.__dict__.pop("originData", None)
        self.__dict__.pop("rawDestinationData", None)
        self.__dict__.pop("destinationData", None)
        self.__dict__.pop("rawOriginData", None)
        self.__erase_traffic_factor_model()
    
    def prepare_traffic_factor_model(self):
        if not self.__trafficFactorModel_class:
            raise ValueError("__trafficFactorModel_class is not specified. Call " 
                             + "`model.set_traffic_factor_model_class(...)`")
            
        self.trafficFactorModel = self.__trafficFactorModel_class(
            self.destinationData, self.originData, self.postalCodeAreaData, 
            self.roadNetwork.shortestDistances, 
            self.roadNetwork.postalCodeDistances)
        
    def fit_flow_model(self, permutations=None, refit=False, 
                       flowParameters=None, continueFlowOptimization=False, 
                       get_CI=True):
        
        self.prst("Fitting flow models.")
        self.increase_print_level()
        
        fittedModel = False
        routeModelData = self.routeModel
        if not refit and "AIC" in routeModelData:
            self.prst("A model does already exist. I skip",
                      "this step. Enforce fitting with the argument",
                      "refit=True")
            return False
        if not "fullCountData" in routeModelData:
            self.prst("The model has no extrapolated boater data. I stop.",
                      "Call preprocess_count_data if you want to",
                      "use the model.")
            return False
        if "trafficFactorModel" not in self.__dict__:
            self.prst("No traffic factor model has been specified. Call "
                      + "model.set_traffic_factor_model(...)!")
            return False
        
        
        self.prst("Fitting the traffic factor model")
        self.increase_print_level()
        
        routeChoiceParams = routeModelData["routeChoiceModel"].parameters
        
        if permutations is None:
            permutations = self.trafficFactorModel.PERMUTATIONS
        
        
        if permutations is None:
            permutations = np.ones(self.trafficFactorModel.SIZE, 
                                    dtype=bool)[None,:]
        
        permutationsFull = []
        for l in permutations:
            permutationsFull.append([True, True] + list(l))
        
        permutations = np.array(permutationsFull)
        parameters = []
        AICs = []
        LLs = []
        results = []
        
        extrapolateCountData = self.routeModel
        if flowParameters is None:
            
            if (not "AIC" in extrapolateCountData and
                    "parameters" in extrapolateCountData 
                    and "covariates" in extrapolateCountData):
                x0 = [extrapolateCountData["parameters"]]
                permutations = np.array([routeModelData["covariates"]])
            else: 
                x0 = repeat(None)
            
            # inspectedRoutes cannot be shared.
            # Therefore, I copy it temporarily
            inspectedRoutes = extrapolateCountData["inspectedRoutes"]
            del extrapolateCountData["inspectedRoutes"]
            
            
            const_args = [extrapolateCountData, self.trafficFactorModel,
                          routeChoiceParams, self.complianceRate, self.properDataRate]
            with ProcessPoolExecutor(const_args=const_args) as pool:
                mapObj = pool.map(
                        HybridVectorModel.maximize_log_likelihood_static,
                        permutations,
                        repeat(3),
                        repeat(None),
                        x0,
                        chunksize=1)
                
                
                for permutation, result in zip(permutations, mapObj):
                    x, xOrig, nLL = result.x, result.xOriginal, result.fun
                    AIC = 2 * (np.sum(permutation) + nLL)
                    AICs.append(AIC)
                    LLs.append(nLL)
                    parameters.append(x)
                    results.append(result)
                    self.prst(permutation, "| AIC:", AIC, 
                              x, xOrig)
            extrapolateCountData["inspectedRoutes"] = inspectedRoutes
            fittedModel = True
        else:
            if continueFlowOptimization:
                result = self.maximize_log_likelihood(
                                  flowParameters["covariates"], 
                                  x0=flowParameters["paramters"])
                parameters = [result.x]
                fittedModel = True
            else:
                result = self.maximize_log_likelihood(
                                  flowParameters["covariates"], 
                                  flowParameters=flowParameters["paramters"])
                parameters = [flowParameters["paramters"]]
            nLL = result.fun
            LLs.append(nLL)
            AICs = [2 * (np.sum(flowParameters["covariates"]) + nLL)]
            permutations = [flowParameters["covariates"]]
            results.append(result)
        
        self.decrease_print_level()
        
        bestIndex = np.argmin(AICs)
        AIC = AICs[bestIndex]
        LL = LLs[bestIndex]
        params = parameters[bestIndex]
        covariates = permutations[bestIndex]
        
        self.prst("Choose the following covariates:")
        self.prst(covariates)
        self.prst("Parameters (transformed):")
        self.prst(params) 
        self.prst("Parameters (original):")
        self.prst(results[bestIndex].xOriginal)
        self.prst("Negative log-likelihood:", LL, "AIC:", AIC)
        
        if results and fittedModel and get_CI: # or True:
            self.investigate_profile_likelihood(params,
                                            extrapolateCountData, 
                                            self.trafficFactorModel, 
                                            routeModelData["routeChoiceModel"].parameters, 
                                            self.complianceRate,
                                            self.properDataRate,
                                            covariates, 
                                            disp=True, vm=False)
                 
            
        if (flowParameters is None or "AIC" not in routeModelData
                or routeModelData["AIC"] > AIC):
            routeModelData["AIC"] = AIC
            routeModelData["parameters"] = params
            routeModelData["covariates"] = permutations[bestIndex]
                    
        self.decrease_print_level()
        return fittedModel
        
    def get_station_mean_variance(self, stationIndices=None,
                                  shiftStart=0, shiftEnd=24,
                                  getStationResults=True,
                                  getPairResults=False,
                                  fullCompliance=False,
                                  correctData=False):
        
        self.prst("Computing the mean and the variance of the traffic at the",
                  "given locations.")
        
        if not getStationResults and not getPairResults: 
            return
        
        if stationIndices is None:
            stationIndices = range(len(
                                    self.roadNetwork.stationIDToStationIndex))
        
        if fullCompliance:
            complianceRate = 1
        else:
            complianceRate = self.complianceRate
        
        if correctData:
            properDataRate = 1
        else:
            properDataRate = self.properDataRate
            
        modelData = self.routeModel
        
        c2, c3, c4 = modelData["routeChoiceModel"].parameters
        
        infested = self.destinationData["infested"]
        
        kMatrix, q = self._get_k_q()
        timeFactor = self.travelTimeModel.interval_probability(shiftStart, shiftEnd
                                                            ) * complianceRate * properDataRate
        
        constantTime = not hasattr(timeFactor, "__iter__")
        
        aq = q * c2 * c4 * timeFactor
        qq = aq/(1-q+aq)
        factor = (1-c2) * q * timeFactor
        addition = c2 * c4 * q * timeFactor
        
        if constantTime:
            blindMeans = kMatrix * (qq / (1-qq))
            blindVariances = blindMeans / (1-qq)
            blindMeanSum = np.sum(blindMeans)
            blindVarianceSum = np.sum(blindVariances)
            blindMeanInfestedSum = np.sum(blindMeans[infested])
            blindVarianceInfestedSum = blindMeanInfestedSum / (1-qq)
            blinds = [blindMeans, blindVariances, blindMeanSum, 
                      blindVarianceSum, blindMeanInfestedSum, 
                      blindVarianceInfestedSum]
            qqBase = repeat(qq)
            factorBase = repeat(factor)
            additionBase = repeat(addition)
        else:
            qqBase = qq
            factorBase = factor
            additionBase = addition 
            blinds = [None] * 6
        
        routeLengthsPowers = copy(modelData["routeLengths"])
        routeLengthsPowers.data = routeLengthsPowers.data.power(c3)
        routeLengthsNorm = routeLengthsPowers.data.sum(1).reshape(kMatrix.shape)
        inspectedRoutes = modelData["inspectedRoutes"]
        stationIDs = self.roadNetwork.stationIndexToStationID
        
        if getStationResults:
            dtype = {"names":["stationID", "mean", "meanInfested", "variance", 
                              "varianceInfested"], 
                     'formats':[IDTYPE, 'double', 'double', 'double', 'double']}
            stationResult = np.zeros(len(stationIndices), dtype=dtype)
        
        if getPairResults:
            dtype = {"names":["mean", "variance"], 
                     'formats':['double', 'double']}
            pairResult = np.zeros(kMatrix.shape, dtype=dtype)
        
        self.increase_print_level()
        counter = Counter(len(stationIndices), 0.01)
        const_args = [routeLengthsPowers, routeLengthsNorm, kMatrix, infested,
                      q, constantTime, getStationResults, getPairResults] + \
                     blinds
        with ProcessPoolExecutor(const_args=const_args) as pool:
            l = [(inspectedRoutes[ind] if ind in inspectedRoutes else {})
                  for ind in stationIndices]
            mapObj = pool.map(
                    HybridVectorModel._get_station_mean_variance_partial,
                    l, qqBase, factorBase, additionBase,
                    chunksize=1)
            
            for i, stationIndex, result in zip(itercount(), stationIndices, 
                                               mapObj):
                percentage = counter.next()
                if percentage: self.prst(percentage, percent=True)
                
                if getStationResults:
                    mean, meanInfested, variance, varianceInfested = result[0]
                    stationResult[i] = (stationIDs[stationIndex], mean, 
                                        meanInfested, variance, 
                                        varianceInfested)
                if getPairResults:
                    means, variances = result[1]
                    pairResult["mean"] += means
                    pairResult["variance"] += variances
        
        self.decrease_print_level()
        
        if getStationResults:
            if getPairResults:
                return stationResult, pairResult
            else:
                return stationResult
        else:
            return pairResult
        
    @staticmethod
    def _get_station_mean_variance_partial(routeLengthsPowers,
                                            routeLengthsNorm,
                                            kMatrix,
                                            infested,
                                            q,
                                            constantTime,
                                            getStationResults,
                                            getPairResults,
                                            blindMeans, 
                                            blindVariances, 
                                            blindMeanSum, 
                                            blindVarianceSum, 
                                            blindMeanInfestedSum, 
                                            blindVarianceInfestedSum,
                                            inspectedRoutesDict, qq, factor,
                                            addition,
                                            ):
        #tedRoutesDict = inspectedRoutes[stationIndex]
        
        if not constantTime:
            #qq = qqBase[i]
            blindMeans = kMatrix * (qq / (1-qq))
            blindVariances = blindMeans / (1-qq)
            #factor = factorBase[i]
            #addition = additionBase[i]
            if getStationResults:
                blindMeanSum = np.sum(blindMeans)
                blindVarianceSum = np.sum(blindVariances)
                blindMeanInfestedSum = np.sum(blindMeans[infested])
                blindVarianceInfestedSum = blindMeanInfestedSum / (1-qq)
        
        if len(inspectedRoutesDict):
            """
            routeProbabilities = [sum(routeLengthsPowers[
                                                    pair[0], pair[1], pathIndex] 
                                      for pathIndex in pathIndices)
                                  / routeLengthsNorm[pair] 
                                  for pair, pathIndices 
                                  in inspectedRoutesDict.items()]
            #"""
            routeProbabilities = [routeLengthsPowers[pair[0], pair[1], pathIndices].sum()
                                  / routeLengthsNorm[pair] 
                                  for pair, pathIndices 
                                  in inspectedRoutesDict.items()]
            
            pairs = tuple(zip(*inspectedRoutesDict.keys()))
            
            aq = np.array(routeProbabilities) * factor + addition
            qq = aq/(1-q+aq)
            qqm = 1 - qq
            
            means = kMatrix[pairs] * qq / qqm
            variances = means / qqm
        else:
            means = variances = 0
        
        if getStationResults and len(inspectedRoutesDict):
            infestedRoutes = [i for i, pair in enumerate(inspectedRoutesDict)
                              if infested[pair[0]]]
            infestedSources = [pairs[0][i] for i in infestedRoutes]
            infestedSinks = [pairs[1][i] for i in infestedRoutes]
            
            mean = np.sum(means) + blindMeanSum - np.sum(blindMeans[pairs])
            variance = (np.sum(variances) + blindVarianceSum 
                        - np.sum(blindVariances[pairs]))
            meanInfested = (np.sum(means[infestedRoutes]) + 
                            blindMeanInfestedSum  
                            - np.sum(blindMeans[infestedSources,
                                                infestedSinks]))
            varianceInfested = (np.sum(variances[infestedRoutes]) 
                                + blindVarianceInfestedSum 
                                - np.sum(blindVariances[infestedSources,
                                                        infestedSinks]))
            
            stationResult = (mean, meanInfested, variance, varianceInfested)
        elif getStationResults:
            stationResult = (blindMeanSum, blindMeanInfestedSum, 
                             blindVarianceSum, blindVarianceInfestedSum)
        else: 
            stationResult = None
        
        if getPairResults:
            pairResultMean = blindMeans
            pairResultVariance = blindVariances
            if  len(inspectedRoutesDict):
                pairResultMean[pairs] += means - blindMeans[pairs]
                pairResultVariance[pairs] += variances - blindVariances[pairs]
            pairResult = (pairResultMean, pairResultVariance)
        else:
            pairResult = None
        
        return stationResult, pairResult
        
    
    def _get_k_q(self, pair=None, shiftStart=None, shiftEnd=None, 
                 stationIndex=None):
        modelData = self.routeModel
        covariates = modelData["covariates"]
        params = self._convert_parameters(modelData["parameters"], covariates)
        q = params[1]
        k = self._get_k_value(params, covariates, pair)
        
        factor = 1
        if stationIndex is not None:
            routeData = self.routeModel
            c2, c3, c4 = routeData["routeChoiceModel"].parameters
            try:
                routeLengths = routeData["routeLengths"][pair].data
                stationPathLengths = [routeLengths[i] for i in 
                                      routeData["inspectedRoutes"][stationIndex][pair]]
                
                stationPathLengths = np.power(stationPathLengths, c3).sum()
                normConstant = np.power(routeLengths, c3).sum()
                factor = (1-c2)*stationPathLengths/normConstant + c2*c4
            except KeyError:
                factor = c2*c4
        
        if shiftStart is not None and shiftEnd is not None:
            timeFactor = self.travelTimeModel.interval_probability(shiftStart, 
                                                                shiftEnd)
            factor *= timeFactor
        
        if not factor == 1:
            q = factor*q/(1-q+factor*q)
        return k, q
    
    
    def optimize_inspection_station_operation(self, 
                costShift,          # costs per inspection shift
                costSite,           # costs per used inspection site
                costBound,          # cost budget
                shiftLength,        # length of shift measured in time steps, 
                                    # if float: approx. length
                nightPremium=None,  #(nightcost, start, end) 
                                    # shift costs in time interval [start, end]
                allowedShifts=None, # list of allowed shift starting times 
                                    # measured in time units
                                    # if floats: approx. shift starting times
                costRoundCoeff=1,   # specifies up to which fraction of the 
                                    # smallest cost the costs are rounded
                baseTimeInv=24,     # number of time intervals, 
                                    # if float: approx. interval
                ignoreRandomFlow=False, # indicates whether we should ignore 
                                        # randomly moving boaters
                integer=True,       # if True, the solver solves the integer 
                                    # problem directly. Otherwise, we use our
                                    # own rounding scheme.
                timeout=1000,
                perturbation=1e-6,
                fancyRounding=True,
                full_result=False,
                extended_info=False,
                init_greedy=True,
                saveFile=True,
                loadFile=True,
                fileNameAddition="",
                ):
        
        # Checking of a result had already been computed earlier
        
        if type(baseTimeInv) != int and type(baseTimeInv) != np.int:
            baseTimeInv = int(round(24 / baseTimeInv))
        baseTime = 24 / baseTimeInv
        if allowedShifts is None:
            allowedShifts = np.arange(baseTimeInv, dtype=int)
        allowedShifts = np.array(allowedShifts)
        if allowedShifts.dtype != np.int:
            allowedShifts = np.round(allowedShifts / baseTime
                                     ).astype(int)
        if type(shiftLength) != int and type(shiftLength) != np.int:
            shiftLength = int(round(shiftLength / baseTime))
        
        # transforming the input
        
        timeIntervalBreaks = np.union1d(allowedShifts, 
                                        (allowedShifts + shiftLength) 
                                        % baseTimeInv)
        timeIntervalStarts = timeIntervalBreaks * baseTime
        timeIntervalEnds = np.roll(timeIntervalBreaks, -1)
        timeIntervalEnds[-1] += baseTimeInv # careful: This time exceeds 24!
        timeIntervalEnds = timeIntervalEnds*baseTime
        
        shiftTimeIntervals = np.vstack((allowedShifts*baseTime, 
                                ((allowedShifts+shiftLength)*baseTime)%24)).T
            
        if costRoundCoeff or not costSite:
            if costRoundCoeff < 1:
                costRoundCoeffInv = round(1/costRoundCoeff)
            else: 
                costRoundCoeffInv = 1/costRoundCoeff
            
            if costShift < costSite or not costSite:
                baseCost = costShift
                costShift = 1
                costSite = (round(costSite / baseCost * costRoundCoeffInv) 
                            / costRoundCoeffInv)
            else:
                baseCost = costSite
                costSite = 1
                costShift = (round(costShift / baseCost * costRoundCoeffInv) 
                             / costRoundCoeffInv)
            
            costBound = (round(costBound / baseCost * costRoundCoeffInv) 
                         / costRoundCoeffInv)
        else: 
            baseCost = 1
        
        if extended_info and not full_result and self.fileName:
            if not os.access(self.fileName, os.F_OK): os.makedirs(self.fileName)
            
            fileName = (self.fileName + fileNameAddition +
                        str([costShift, costSite, costBound, shiftLength,
                             nightPremium, int(ignoreRandomFlow)]) + ".dat")
            fileName = os.path.join(self.fileName, fileName)
            if loadFile and exists(fileName):
                return saveobject.load_object(fileName)
        else:
            saveFile = False
            
        self.prst("Optimizing inspection station placement",
                  "and operating times.")
        self.increase_print_level()
        
        self.prst("Using the following parameters:")
        self.increase_print_level()
        self.prst("Cost per shift:", baseCost*costShift)
        self.prst("Cost per site:", baseCost*costSite)
        self.prst("Budget:", baseCost*costBound)
        self.prst("Length of the shifts:", baseTime*shiftLength)
        self.prst("Shift start times:", baseTime*allowedShifts)
        self.prst("Interval start times:", timeIntervalStarts)
        self.prst("Interval end times:", timeIntervalEnds)
        self.prst("Time unit:", baseTime)
        self.prst("Cost unit:", baseCost)
        self.prst("Perturbation:", perturbation)
        self.prst("Ignore random boater flow:", ignoreRandomFlow)
        self.decrease_print_level()
        
        # prepare problem matrices and vectors
        
        modelData = self.routeModel
        
        stationCombinations = modelData["stationCombinations"]
        stationIndexToRelevantStationIndex = {spot:i for i, spot in 
                                              enumerate(
                                                  modelData["inspectedRoutes"
                                                            ].keys())}
        relevantStationIndexToStationIndex = np.zeros(
                            len(stationIndexToRelevantStationIndex), dtype=int)
        
        for spot, i in stationIndexToRelevantStationIndex.items():
            relevantStationIndexToStationIndex[i] = spot
        
        originInfested = self.destinationData["infested"]
        covariates = modelData["covariates"] 
        params = self._convert_parameters(modelData["parameters"], covariates)
        routeLengths = modelData["routeLengths"].data
        
        q = params[1]
        c2, c3, c4 = modelData["routeChoiceModel"].parameters
        
        
        intervalTimeFactors = self.travelTimeModel.interval_probability(
                                        timeIntervalStarts, timeIntervalEnds)
        shiftTimeFactors = self.travelTimeModel.interval_probability(
                                        baseTime*allowedShifts, 
                                        baseTime*(allowedShifts+shiftLength))

        
        routePowers = routeLengths.power(c3)
        routeNormConsts = np.asarray(routePowers.sum(1)).ravel()
        
        stationArrI = []
        stationArrJ = []
        sinkNumber = self.roadNetwork.shortestDistances.shape[1]
        
        kArray = self._get_k_value(params, covariates)
        
        kArrayInfested = kArray[self.destinationData["infested"]]
        totalRandomFlow = np.sum(kArrayInfested) * c2 * q / (1-q)
        
        kArray = kArray.ravel()
        
        # List of full-day flows
        fullFlowList = []
        
        # iterate over path sets with distinct station sets
        for stationSet, flowInfos in stationCombinations.items():
            
            if not len(stationSet):
                continue
            
            # contains the pair index and the flow index for all considered 
            # flows
            tmpArr = np.array([(sinkNumber*source + sink, flowIndex) 
                               for source, sink, flowIndex in flowInfos
                               if originInfested[source]], dtype=int)
            
            if len(tmpArr):
                i = len(fullFlowList)
                
                pairIndices, flowIndices = tmpArr.T
                
                # proportional to full-day flow
                flow = np.sum(kArray[pairIndices]
                              * np.asarray(routePowers[pairIndices, flowIndices]
                                           ).ravel()
                              / routeNormConsts[pairIndices]) * (q / (1-q) * (1-c2))
                
                fullFlowList.append(flow)
                
                # note the flow-station pairs that are related
                # Y_ij = 1, iff stationArrI[k]=i and stationArrJ[k]=1 for some k
                stationArrI.extend([i] * len(stationSet))
                stationArrJ.extend(
                        stationIndexToRelevantStationIndex[stationIndex] for
                        stationIndex in stationSet)
            
        flowNumber = len(fullFlowList)
        spotNumber = len(stationIndexToRelevantStationIndex)
        timeNumber = len(intervalTimeFactors)
        shiftNumber = len(shiftTimeFactors)
        
        self.prst("We optimize {} flows at {} times over {} locations and {} shifts.".format(
                    flowNumber, timeNumber, spotNumber, shiftNumber))
        
        flowDim = flowNumber * timeNumber
        inspectionDim = shiftNumber * spotNumber
        
        fullFlowList = np.array(fullFlowList)
        
        intervalFlows = fullFlowList[:,None] * (intervalTimeFactors*self.complianceRate)
        flowCoefficients = intervalFlows.ravel()
        totalPredictableFlow = np.sum(kArrayInfested)*q/(1-q)*(1-c2)
        totalFlow = (totalPredictableFlow 
                     + (not ignoreRandomFlow)*totalRandomFlow)
        
        self.prst("The total reducable predictable flow is {}.".format(
                                                        np.sum(intervalFlows)))
        self.prst("The total predictable flow is {}.".format(
                                                        totalPredictableFlow))
        self.prst("The total random flow is {}.".format(totalRandomFlow))
        self.prst("Creating subSubMatrixTime")
        
        # subMatrixTime[i, j] = 1, iff shift j covers time interval i
        subSubMatrixTime = np.zeros((timeNumber, shiftNumber), dtype=int)
        for i, t in enumerate(timeIntervalBreaks):
            for j, tStart, tEnd in zip(itercount(), allowedShifts, 
                                       (allowedShifts+shiftLength) 
                                                                % baseTimeInv):
                if tStart < tEnd:
                    if t >= tStart and t < tEnd:
                        subSubMatrixTime[i, j] = 1
                else:
                    if t >= tStart or t < tEnd:
                        subSubMatrixTime[i, j] = 1
        
        def setSubmatrix(spm, pos, subm, grid=None):
            if grid is None:
                dim1, dim2 = subm.size
                i, j = pos
                spm[i:i+dim1, j:j+dim2] = subm
            else:
                i, j = pos
                fact1, fact2 = grid
                spm[i*fact1:(i+1)*fact1, j*fact2:(j+1)*fact2] = subm
        
        
        self.prst("Creating matrixFlowConstraint")
        matrixFlowConstraint = sparse.dok_matrix((flowDim, inspectionDim))
        for i, j, k in zip(stationArrI, stationArrJ, itercount()):
            setSubmatrix(matrixFlowConstraint, (i,j), subSubMatrixTime, 
                         subSubMatrixTime.shape)
            if not k % 500:
                self.prst(round(k/len(stationArrI)*100,2))
        
        self.prst("Creating matrixTimeConstraint")
        matrixTimeConstraint = sparse.dok_matrix((spotNumber*timeNumber, 
                                                       inspectionDim))
        for i in range(spotNumber):
            setSubmatrix(matrixTimeConstraint, (i,i), subSubMatrixTime,
                         subSubMatrixTime.shape)
            if not i % 10:
                self.prst(round(i/spotNumber*100,2))
        
        if integer:
            operations = cp.Variable(inspectionDim, boolean=True)
            spotusage = cp.Variable(spotNumber) #, boolean=True)
            flows = cp.Variable(flowDim) #, integer=True)
        else:
            flows = cp.Variable(flowDim)
            spotusage = cp.Variable(spotNumber)
            operations = cp.Variable(inspectionDim)
        #operations = cp.Variable(inspectionDim, integer=True)
        
        if integer and init_greedy:
            self.prst("Applying greedy rounding to determine good initial guess")
            flows_, operations_, spotusage_ = self.optimize_inspection_station_operation(
                costShift, costSite, costBound, shiftLength, nightPremium, 
                allowedShifts, costRoundCoeff, baseTimeInv, ignoreRandomFlow, 
                False, timeout, perturbation, fancyRounding, True)[1]
            flows.value = flows_
            spotusage.value = spotusage_
            operations.value = operations_
            warm_start = True
        else:
            warm_start = False
            spotusageConstr = np.zeros(spotusage.size)
            operationConstr = np.zeros(operations.size)
        
        
        #coveredFlow = flowCoefficients@flows + operationCoefficients@operations
        
        if nightPremium:
            costShiftNight, nightStart, nightEnd = nightPremium
            transformedShiftStart = allowedShifts.copy()
            transformedShiftStart[transformedShiftStart<nightEnd] += 24
            if nightStart < nightEnd:
                nightStart += 24
            nightEnd += 24
            left = np.maximum(transformedShiftStart, nightStart)
            right = np.minimum(transformedShiftStart+shiftLength, nightEnd)
            factor = np.maximum(right-left, 0)/shiftLength
            costShift_ = factor*costShiftNight + (1-factor)*costShift
            costShift__ = np.tile(costShift_, spotNumber)
        else:
            costShift__ = costShift
            
        coveredReducableFlow = flowCoefficients@flows
        
        if ignoreRandomFlow:
            coveredFlow = coveredReducableFlow
        else:
            randomFlows = np.sum(kArrayInfested) * shiftTimeFactors * (c2*c4*q/(1-q)*self.complianceRate)
            operationCoefficients = np.tile(randomFlows, spotNumber)
            coveredRandomFlow = cp.minimum(operationCoefficients@operations, 
                                           totalRandomFlow*self.complianceRate)
            coveredFlow = coveredReducableFlow + coveredRandomFlow
            
            #coveredFlow = flowCoefficients@flows + operationCoefficients@operations
        if perturbation:
            coveredFlow += cp.sum(operations[::shiftNumber])*perturbation
                
        self.prst("Creating cost constraint")
        
        remainingBudget = (costBound - cp.sum(operations * costShift__)
                                     - cp.sum(spotusage) * costSite)
        costConstraint = [remainingBudget >= 0]
        
        """
        update locOperated to exclude locations that have been chosen to be used. 
        Determine the best location first, then set it to be chosen, 
        reduce the remaining budget but remove the location cost for this location
        
        determine for each station the maximal usage value. For the station with the maximal
        usage value remove the cost constraint, if not done already.
        if done already, register highest value.
        #"""
        operatingTimeSums = matrixTimeConstraint*operations
        
        constraints = [
                flows <= matrixFlowConstraint*operations,
                flows <= 1,
                operations >= 0,
                #spotusage <= 1
                ]
        
        constraints += [operatingTimeSums[i::timeNumber] <= spotusage for i
                        in range(timeNumber)]
        
        equalityConstraints = []
        coveredFlowValues = []
        
        #co.solvers.options['interior'] = False
        locationUsed = np.zeros(spotNumber, dtype=bool)
        iterations = 0
        noNewSpot = False
        relaxAffordable = True
        fixedLocations = []
        success = True
        repetition = False
        while True:
            iterations += 1
            self.prst("Creating optimizer")
            optimizer = cp.Problem(cp.Maximize(coveredFlow), costConstraint
                                   +constraints+equalityConstraints)
            if integer:
                self.prst("Solving ILP")
            else:
                self.prst("Solving LP")
            
            try:
                #if integer:
                #    optimizer.solve(verbose=True, warm_start=warm_start, solver=cp.SCS)
                #else:
                optimizer.solve(verbose=True, warm_start=warm_start, solver=cp.MOSEK, 
                                mosek_params=dict(MSK_DPAR_MIO_MAX_TIME=timeout,
                                                  MSK_DPAR_MIO_TOL_REL_GAP=0.005))
            except cp.error.SolverError as e:
                print("Solver error", e)
                pass
            warm_start = True
            
            if optimizer.status == "optimal" or optimizer.solver_stats.solve_time>=timeout: 
                coveredFlowValues.append(coveredFlow.value)
                self.prst("Optimization was successful. Covered flow:",
                          "{} ({:6.2%} of optimal value)".format( 
                              coveredFlowValues[-1], 
                              coveredFlowValues[-1]/coveredFlowValues[0]))
                
                operatingTimeSumsOriginal = np.array(operatingTimeSums.value).round(4)
                operationResult = np.array(operations.value).round(4)
                operations.value = operationResult
                operationResultReshaped = operationResult.reshape((spotNumber,shiftNumber))
                if not ignoreRandomFlow:
                    coveredRandomFlowCorrect = (1-np.prod(1-np.sum(operationResultReshaped*shiftTimeFactors, 1)
                                                  *c4))*totalRandomFlow*self.complianceRate
                    self.prst(("Random flow: {:6.2f} ({:6.2%} of total covered flow; "
                               "actual random flow: {:6.2f} ({:6.2%} overestimate)"
                               ).format(coveredRandomFlow.value,
                                        coveredRandomFlow.value/coveredFlow.value,
                                        coveredRandomFlowCorrect,
                                        (coveredRandomFlow.value-coveredRandomFlowCorrect)/
                                        coveredRandomFlowCorrect
                                        ))
                
                #print("spotusage", np.round(np.array(spotusage.value).ravel(), 3))
                #print("flows", np.round(np.array(flows.value).ravel(), 3).reshape((flowNumber, timeNumber)))
                
                print(operationResultReshaped[np.any(operationResultReshaped > 0, 1)])
                print(np.array(spotusage.value)[np.any(operationResultReshaped > 0, 1)])
                
                if ((operationResult == 1) | (operationResult == 0)).all():
                    
                    # just because there could be multiple feasible solutions
                    spotusage.value = np.array(operations.value).reshape((spotNumber, shiftNumber)).max(1)
                    self.prst("Reached integer solution")
                    
                    break
                
                operations.value = (operationResult==1).astype(int)
                spotusageOriginal = np.array(spotusage.value).round(4)
                spotusage.value = np.array(operations.value).reshape((spotNumber, shiftNumber)).max(1)
                
                
                # ---------------
                oss = np.array(operatingTimeSums.value)
                sp = np.array(spotusage.value)
                viol = np.array([oss[i::timeNumber] > sp for i
                        in range(timeNumber)])
                #print("viol", viol)
                print("viol", viol.sum())
                # ---------------
                
                remainingBudgetValue = remainingBudget.value
                
                locationUsed |= np.array(spotusage.value).astype(bool).ravel()
                locationUsed_ = np.tile(locationUsed, (shiftNumber, 1)).T.ravel()
                affordable = (costShift__+(~locationUsed_)*costSite <= remainingBudgetValue)
                #covered = np.array(operatingTimeSums.value).astype(bool)
                covered = (matrixTimeConstraint.T@operationConstr + matrixTimeConstraint@operationConstr).astype(bool)
                mask = covered | (~affordable) | np.array(operations.value).astype(bool)
                #print(covered.reshape((spotNumber, shiftNumber))[np.any(operationResultReshaped > 0, 1)])
                #print(affordable.reshape((spotNumber, shiftNumber))[np.any(operationResultReshaped > 0, 1)])
                nonIntegralSpot = (1e-4 < spotusageOriginal) & (spotusageOriginal < 0.999)
                #nonIntegralSpotCosts = (spotusageOriginal[nonIntegralSpot].sum()*costSite
                #                        +(operationResultReshaped[nonIntegralSpot]*costShift_).sum())
                nonIntegralShift = (1e-4 < operatingTimeSumsOriginal) & (operatingTimeSumsOriginal < 0.999)
                """
                addCostSpot = (spotusageOriginal-spotusage.value).round(4).any() * costSite
                print(remainingBudgetValue, addCostSpot,
                      2*costShift+addCostSpot-0.0001,
                      noNewSpot, nonIntegralSpot.any(), constrainNonIntegralShifts, 
                      nonIntegralShift.any(), )
                """
                
                if mask.all():
                    if noNewSpot:
                        fixedLocations.append(np.nonzero(nonIntegralShift.reshape((spotNumber,timeNumber)).any(1))[0][0])
                    noNewSpot = True
                    repetition = False
                    #if not nonIntegralSpot.any(): # and nonIntegralSpotCosts >= costShift:
                        #self.prst("No admissible location is affordable")
                        #break
                #elif fancyRounding and (
                #        (remainingBudgetValue < 2*costShift+addCostSpot-0.0001 
                #       and (noNewSpot or not nonIntegralSpot.any())
                #       and nonIntegralShift.any())
                #       or constrainNonIntegralShifts):
                #    constrainNonIntegralShifts = True
                # if no variable that should be constrained to 0 is positive
                elif not (mask & (0 < operationResult))[operationResult<1].any():
                    
                    argmax = np.argmax(np.ma.array(operationResult, mask=mask))
                    maxLocation = argmax // shiftNumber
                    self.prst("Rounding up location", self.roadNetwork.stationIndexToStationID[
                                                        relevantStationIndexToStationIndex[
                                                           maxLocation]])
                    if spotusageOriginal[maxLocation] > 0.999:
                        intervalCoverage = operatingTimeSums.value[maxLocation*timeNumber:(maxLocation+1)*timeNumber]
                        newIntervalCoverage = np.ma.array(operatingTimeSumsOriginal[maxLocation*timeNumber:(maxLocation+1)*timeNumber],
                                                          mask=intervalCoverage.astype(bool))
                        if noNewSpot or not nonIntegralSpot.any():                                              
                            while True:
                                firstNewIntervalCovered = np.argmax(newIntervalCoverage)
                                shiftFirstNewIntervalCoverd = np.nonzero(subSubMatrixTime[:,firstNewIntervalCovered])[0][0]
                                newIndex = int(shiftNumber*maxLocation + shiftFirstNewIntervalCoverd)
                                if not mask[newIndex]:
                                    break
                                else:
                                    newIntervalCoverage.mask[shiftFirstNewIntervalCoverd] = True
                        else:
                            newIndex = int(argmax) # because np.int64 is not supported
                        operations.value[newIndex] = 1
                        operationConstr[newIndex] = 1
                        self.prst("Rounding up interval [{}, {}]".format(*shiftTimeIntervals[newIndex % shiftNumber]))
                    
                    locationUsed[maxLocation] = True
                    spotusageConstr[maxLocation] = 1
                    spotusage.value = locationUsed.astype(int)
                    locationUsed_ = np.tile(locationUsed, (shiftNumber, 1)).T.ravel()
                    affordable = (costShift__+(~locationUsed_)*costSite <= remainingBudget.value)
                    if ((noNewSpot or not nonIntegralSpot.any()) and not
                        affordable.reshape((spotNumber, shiftNumber))[maxLocation].any()):
                        fixedLocations.append(maxLocation)
                        if len(fixedLocations) == locationUsed.sum():
                            self.prst("All locations fixed.")
                            break
                    
                    if not (affordable & (operatingTimeSums.value<1)).any():
                        #if (nonIntegralSpot & (~locationUsed)).any(): # and 
                        #    #nonIntegralSpotCosts >= costShift):
                        noNewSpot = True
                        #else:
                        #    self.prst("No admissible location is affordable after rounding")
                        #    break
                    if not remainingBudget.value:
                        self.prst("The budget has been used completely.")
                        break
                    
                    chosenLocations = np.nonzero(locationUsed)[0]
                    chosenOperations = np.array(operations.value, dtype=bool
                                                ).reshape((spotNumber, 
                                                           shiftNumber))
                    self.prst("Chose the following spots so far:") 
                    self.increase_print_level()
                    usedStationIDs = self.roadNetwork.stationIndexToStationID[
                                            relevantStationIndexToStationIndex[
                                                               chosenLocations]]
                    order = np.argsort(usedStationIDs)
                    for i, iD in zip(chosenLocations[order], usedStationIDs[order]):
                        self.prst(iD, *shiftTimeIntervals[chosenOperations[i]],
                                  ("*" if i in fixedLocations else ""))
                    self.decrease_print_level()
                    
                    self.prst("Remaining budget:", remainingBudget.value * baseCost)
                    repetition = False
                else:
                    if repetition:
                        #fixedLocations.append(np.nonzero((mask & (0 < operationResult))[operationResult<1] // shiftNumber)[0][0])
                        fixedLocations.append(np.nonzero((mask & (0 < operationResult))[operationResult<1])[0][0] // shiftNumber)
                        repetition = False
                    else:
                        repetition = True
                #operatingTimeSumsReshaped = np.array(operatingTimeSums.value).reshape((spotNumber, shiftNumber)) 
                
                #print(operatingTimeSumsReshaped[np.any(operationResultReshaped > 0, 1)])
                
                if noNewSpot and relaxAffordable:
                    operationConstr[:] = 0
                if noNewSpot: 
                    affordable[:] = True
                
                if fixedLocations:
                    if len(fixedLocations) == locationUsed.sum():
                        self.prst("All locations fixed.")
                        break
                    
                    self.prst("Fixate spot usage")
                    fixedOperation = np.zeros((spotNumber, shiftNumber), dtype=bool)
                    fixedOperation[fixedLocations] = True
                    fixed = fixedOperation.ravel()
                    operationConstr[fixed] = np.array(operations.value)[fixed]
                    #equalityConstraints += [operatingTimeSums[i*timeNumber:(i+1)*timeNumber] 
                    #                            == operatingTimeSums.value[i*timeNumber:(i+1)*timeNumber]
                    #                            for i in fixedLocations]
                else:
                    fixed = ~affordable
                
                covered = (matrixTimeConstraint.T@operationConstr + matrixTimeConstraint@operationConstr).astype(bool)
                previous = False
                equalityConstraints = []
                    
                for i, v, c, a in zip(itercount(), operationConstr, covered, 
                                      fixed):
                    if c or a:
                        if not previous:
                            index = i
                            constrArr = []
                        constrArr.append(v)
                        previous = True
                    else:
                        if previous:
                            equalityConstraints.append(operations[index:i] == np.array(constrArr))
                        previous = False
                if previous:
                    equalityConstraints.append(operations[index:] == np.array(constrArr))
                
                #equalityConstraints = [operations[i] == v for i, v, c, a in
                #                       zip(itercount(), operations.value, 
                #                           operatingTimeSums.value, affordable)
                #                       if c or not a]
                
                # just for debugging
                #equalityConstraints_ = [(self.roadNetwork.stationIndexToStationID[
                #                                        relevantStationIndexToStationIndex[
                #                                           i // shiftNumber]],
                #    i  % shiftNumber,
                #    operations.value[i], v) for i, v, c, a in
                #                       zip(itercount(), operations.value, 
                #                           operatingTimeSums.value, affordable)
                #                       if c or not a]
                """
                if constrainNonIntegralShifts:
                    self.prst("Rounding shift choice")
                    #TODO make this wrap around the day
                    newNonIntegralShiftSpot = np.nonzero(nonIntegralShift.reshape((spotNumber,timeNumber)).any(1))[0][0]
                    nonIntegralShiftSpots = np.unique(np.concatenate([newNonIntegralShiftSpot], nonIntegralShiftSpots)).astype(int)
                    for i in nonIntegralShiftSpots:
                        thisSpotOperation = operations.value[i*timeNumber:(i+1)*timeNumber]
                        coveredIntervals = subSubMatrixTime@thisSpotOperation
                        roundedIntervals = np.floor(operatingTimeSumsOriginal)[i*timeNumber:(i+1)*timeNumber]
                        newCoveredIntervals = (roundedIntervals.astype(bool) 
                                               & (~coveredIntervals.astype(bool)))
                        firstNewIntervalCovered = np.nonzero(newCoveredIntervals)[0][0]
                        #shiftFirstNewIntervalCoverd = np.nonzero(subSubMatrixTime[:,firstNewIntervalCovered])[0][0]
                        equalityConstraints.append(operatingTimeSums[i*timeNumber:i*timeNumber+firstNewIntervalCovered+1]
                                                   == newCoveredIntervals[:firstNewIntervalCovered+1])
                        
                    
                    equalityConstraints += [operatingTimeSums[i*timeNumber:(i+1)*timeNumber] 
                                            == np.floor(operatingTimeSumsOriginal)[i*timeNumber:(i+1)*timeNumber]
                                            for i in nonIntegralShiftSpots]
                    spotusage.value = spotusageOriginal
                    noNewSpot = True
                """
                if noNewSpot:
                    # fixate spotusage
                    self.prst("Fixate spot choice")
                    equalityConstraints += [spotusage == spotusage.value]
                    relaxAffordable = False
                else:
                    equalityConstraints += [spotusage[i] == 1 for i, v in
                                            enumerate(spotusage.value) if v > 0.999]
                
                    
                #y = matrixFlowConstraint*operations.value
                #equalityConstraints += [flows[i] == 1 for i, v in
                #                        enumerate(y) if v]
            else:
                success = False
                break
        
        if not success:
            self.prst("Optimization failed with message:", optimizer.status)
            return
        
        self.prst("Optimization process terminated successfully",
                  "after {} iteration(s).".format(iterations))
        y = matrixFlowConstraint*operations.value
        flows.value = np.array([min(yy, 1) for yy in y])
        coveredFlowValue = coveredFlow.value
        operationResult = np.array(operations.value).reshape((spotNumber, 
                                                              shiftNumber))
        operationResult = np.round(operationResult, 4).astype(bool)
        chosenLocations = np.nonzero(np.round(spotusage.value, 4))[0]
        self.increase_print_level()
        
        if integer:
            intGap = coveredFlowValue/coveredFlowValues[0]
        else:
            intGap = coveredFlowValue/coveredFlowValues[0]
        
        self.prst("Covered flow:    {} ({:6.2%} of optimal value; {:6.2%} of total flow)".format( 
                              coveredFlowValue, 
                              intGap,
                              coveredFlowValue/totalFlow))
        if not ignoreRandomFlow:
            coveredRandomFlowCorrect = (1-np.prod(1-np.sum(operationResult*shiftTimeFactors, 1)
                                                  *c4))*totalRandomFlow*self.complianceRate
            self.prst(("Random flow: {:6.2f} ({:6.2%} of total covered flow; "
                               "actual random flow: {:6.2f} ({:6.2%} overestimate)"
                               ).format(coveredRandomFlow.value,
                                        coveredRandomFlow.value/coveredFlow.value,
                                        coveredRandomFlowCorrect,
                                        (coveredRandomFlow.value-coveredRandomFlowCorrect)/
                                        coveredRandomFlowCorrect
                                        ))
        self.prst("Remainig budget: {}".format(remainingBudget.value * baseCost))
        self.prst("Chose the following spots:") 
        self.increase_print_level()
        usedStationIDs = self.roadNetwork.stationIndexToStationID[
                            relevantStationIndexToStationIndex[chosenLocations]]
        order = np.argsort(usedStationIDs)
        for i, iD in zip(chosenLocations[order], usedStationIDs[order]):
            self.prst(iD, *shiftTimeIntervals[operationResult[i]])
        self.decrease_print_level()
        self.decrease_print_level()
        self.decrease_print_level()
        
        if ignoreRandomFlow:
            coveredFlowCorrect = coveredFlowValue
        else:
            coveredFlowCorrect = coveredRandomFlowCorrect + coveredReducableFlow.value
            self.prst("Corrected total flow: {:6.2f} ({:6.2%} error)".format(
                coveredFlowCorrect, (coveredFlowValue-coveredFlowCorrect)/coveredFlowCorrect))
        
        # ---------------
        oss = np.array(operatingTimeSums.value)
        sp = np.array(spotusage.value)
        viol = np.array([oss[i::timeNumber] > sp for i
                in range(timeNumber)])
        #print("viol", viol)
        print("viol", viol.sum())
        # ---------------
        
        result = coveredFlowCorrect
        
        
        if extended_info:
            dtype = [("stationID", IDTYPE), ("flowCover", float), ("timeCover", float)]
            
            info = np.zeros_like(usedStationIDs, dtype=dtype)
            info["stationID"] = usedStationIDs
            for i, stationIndex in enumerate(chosenLocations):
                ops = np.zeros_like(operationResult, dtype=float)
                ops[stationIndex] = 1
                info["flowCover"][i] = np.sum(fullFlowList[(matrixFlowConstraint@ops.ravel()).reshape((flowNumber, timeNumber)).max(1) > 0])
                info["timeCover"][i] = np.sum(operationResult[stationIndex]
                                              *shiftTimeFactors)
            result = result, coveredFlowCorrect/totalFlow, info
        
        #print("returning", np.array(operations.value).reshape((spotNumber, shiftNumber))[np.array(operations.value).reshape((spotNumber, shiftNumber)).max(1) > 0])
        if full_result:
            result = result, (np.array(flows.value), np.array(operations.value),
                              np.array(spotusage.value))
        
        if saveFile:
            saveobject.save_object(result, fileName)
        
        return result
    
    def create_caracteristic_plot(self, characteristic, values, 
                                  characteristicName=None, valueNames=None,
                                  **optim_kwargs):
        
        optim_kwargs.pop("characteristic", None)
        optim_kwargs.pop("full_result", None)
        optim_kwargs.pop("extended_info", None)
        
        if not characteristicName:
            characteristicName = str(characteristic)
        
        figsize = (4, 3)
        rect = [0.17, 0.18, 0.75, 0.78]
        ax = plt.figure(figsize=figsize).add_axes(rect)
        plt.locator_params(nbins=4) #nticks=3, 
        
        markers = ["o", "x", ".", "v", "s", "p", "D", ">", "*", "X"]
        
        if valueNames is None:
            valueNames = values
        if not hasattr(characteristic, "__call__"):
            optim_kwargs.pop(characteristic, None)
        
        for val, m, name in zip(values, markers, valueNames):
            name = str(name)
            if hasattr(characteristic, "__call__"):
                kwargs = optim_kwargs
                characteristic(self, val)
                fileNameAddition = characteristicName[:3] + name[:5]
            else:
                kwargs = {characteristic:val, **optim_kwargs}
                fileNameAddition = ""
                
            _, _, info = self.optimize_inspection_station_operation(
                full_result=False, extended_info=True, 
                fileNameAddition=fileNameAddition, **kwargs)
            
            if m == ".":
                mfc = {}
            else:
                mfc = dict(markerfacecolor="None")
            
            print("Covered flow", name, info["flowCover"])
            
            ax.plot(info["timeCover"], info["flowCover"], linestyle="",
                     marker=m, markersize=10,
                        label=name, **mfc)
        
        
        ax.set_xlabel("Fraction of daily traffic covered")
        ax.set_ylabel("Daily boater traffic")
        ax.set_yscale("log")
        ax.set_ylim((0.1, 3))
        
        l = plt.legend(title=characteristicName, fancybox=True, loc='upper left', framealpha=0.5) #frameon=False loc='upper left', 
        #l.get_frame().set_linewidth(0.0)
        
        
        
        fileName = self.fileName + "Char_" + characteristicName
        
        plt.savefig(fileName + ".pdf")
        plt.savefig(fileName + ".png", dpi=1000)
        plt.show()
            
            
        
    
    def create_budget_plots(self, minBudget, maxBudget, nSteps=10, 
                            **optim_kwargs):
        figsize = (4, 3)
        budgets = np.linspace(minBudget, maxBudget, nSteps)
        fractions = np.zeros_like(budgets)
        boaters = np.zeros_like(budgets)
        
        optim_kwargs.pop("costBound", None)
        optim_kwargs.pop("full_result", None)
        optim_kwargs.pop("extended_info", None)
        
        for i, budget in enumerate(budgets):
            coveredFlow, coveredFraction, _ = self.optimize_inspection_station_operation(
                costBound=budget, full_result=False, extended_info=True, 
                **optim_kwargs)
            fractions[i] = coveredFraction
            boaters[i] = coveredFlow
        
        print("Budgets", budgets)
        print("Covered Fractions", fractions)
        print("Boaters", boaters)
        print("Price", budgets/boaters)
        
        rect = [0.15, 0.18, 0.75, 0.78]
        ax = plt.figure(figsize=figsize).add_axes(rect)
        ax.set_xlabel('Budget')
        ax.set_ylabel('Fraction of boaters inspected')
        ax.plot(budgets, fractions)
        ax.plot(budgets, np.ones_like(budgets)*self.complianceRate, ":k")
        ax.set_ylim((0, 1))
        plt.locator_params(nbins=4) #nticks=3, 
        #plt.tight_layout()
        
        fileName = self.fileName + "Budget" + str([minBudget, maxBudget])
        plt.savefig(fileName + ".pdf")
        plt.savefig(fileName + ".png", dpi=1000)
        
        rect = [0.15, 0.18, 0.75, 0.78]
        ax = plt.figure(figsize=figsize).add_axes(rect)
        ax.set_ylabel('Costs per inspected high-risk boater')
        ax.set_xlabel('Fraction of boaters inspected')
        ax.plot(fractions, budgets/boaters, "-")
        ax.set_ylim(bottom=0)
        plt.locator_params(nbins=4) #nticks=3, 
        
        #plt.tight_layout()
        
        fileName = self.fileName + "Price" + str([minBudget, maxBudget])
        plt.savefig(fileName + ".pdf")
        plt.savefig(fileName + ".png", dpi=1000)
        
        """
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Budget')
        ax1.set_ylabel('Fraction of boaters inspected')
        ax1.plot(budgets, fractions)
        ax1.plot(budgets, np.ones_like(budgets)*self.complianceRate, ":k")
        ax1.set_ylim((0, 1))
        ax1.locator_params(nticks=3, nbins=3)
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Price per boater')
        ax2.plot(budgets, budgets/boaters, "-", color="C1")
        ax2.locator_params(nticks=3, nbins=3)
        fig.tight_layout()
        """
        
        
        plt.show()
        
        
            
        
        
        
        
        
    def get_pair_distribution_measure(self, measure=nbinom.mean, arg=None, 
                                      pair=None, shiftStart=None,
                                      shiftEnd=None):
        k, q = self._get_k_q(pair, shiftStart, shiftEnd)
        qm = 1-q
        if arg is None:
            if hasattr(measure, "__iter__"):
                return [m(k, qm) for m in measure]
            else:
                return measure(k, qm)
        else:
            if hasattr(measure, "__iter__"):
                if hasattr(arg, "__iter__"):
                    return [m(a, k, qm) if a is not None else m(k, qm) 
                            for m, a in zip(measure, arg)]
                else:
                    return [m(arg, k, qm) for m in measure]
            else:
                return measure(arg, k, qm)
    
    def get_station_observation_prediction(self, predictions=None):
        countData = self.shiftData
            
        if predictions is None:
            rawPredictions = self.get_station_mean_variance(
                    countData["stationIndex"], countData["shiftStart"], 
                    countData["shiftEnd"])
        else: 
            rawPredictions = predictions
        #                            count, mean, variance
        predictions = defaultdict(lambda: [0, 0, 0])
        
        for stationID, count, mean, variance in zip(rawPredictions["stationID"], 
                                                    countData["totalCount"],
                                                    rawPredictions["mean"], 
                                                    rawPredictions["variance"]):
            item = predictions[stationID]
            item[0] += count
            item[1] += mean
            item[2] += variance
        
        dtype = {"names":["stationID", "count", "mean", "variance"], 
                 'formats':[IDTYPE, int, 'double', 'double']}
        
        return np.array([(iD, *vals) for iD, vals in predictions.items()], 
                        dtype=dtype)
    
    def get_normalized_observation_prediction(self, minSampleSize=20):
        
        countData = self.shiftData
        countData = countData[countData["prunedCount"] >= 0]
        
        # use only observations with a sufficiently large sample
        stationIndices, reverse_indices, counts = np.unique(countData["stationIndex"], 
                                                            return_inverse=True, 
                                                            return_counts=True)
        considered = counts >= minSampleSize
        stationIndices = stationIndices[considered]
        
        resultDType = [("stationID", IDTYPE),
                       ("observed", float),
                       ("predicted", float),
                       ("observedMean", float),
                       ("observedMeanScaled", float),
                       ("predictedMeanScaled", float)]
        
        if not len(stationIndices):
            return np.zeros(0, dtype=resultDType)
        
        counts = counts[considered]
        
        rawPredictions = self.get_station_mean_variance(
                stationIndices, self.pruneStartTime, 
                self.pruneEndTime)
        
        resultData = np.zeros(len(stationIndices), dtype=resultDType)
        resultData["stationID"] = rawPredictions["stationID"]
        
        for i, station in enumerate(np.nonzero(considered)[0]):
            resultData["observed"][i] = np.sum(countData["prunedCount"][
                                                    reverse_indices==station])
            resultData["observedMean"][i] = np.mean(countData["prunedCount"][
                                                    reverse_indices==station])
        
        # get the observed and predicted mean values
        resultData["predicted"] = rawPredictions["mean"]
        
        # get the predicted varience (note: By CLT, the variance decreases 
        #                                   with the number of samples)
        # scale the mean values to have a variance of 1 so that they become 
        # comparable (note: By CLT they are approximately normal distributed)
        resultData["observed"] -= counts*rawPredictions["mean"]
        resultData["observed"] /= np.sqrt(rawPredictions["variance"] * counts)
        resultData["observed"] += rawPredictions["mean"]
        stdFact = np.sqrt(counts / rawPredictions["variance"])
        resultData["observedMeanScaled"] = resultData["observedMean"] * stdFact
        resultData["predictedMeanScaled"] = rawPredictions["mean"] * stdFact
        
        return resultData
    
    def get_pair_observation_prediction(self, predictions=None):
        if predictions is None:
            countData = self.shiftData
            predictions = self.get_station_mean_variance(
                    countData["stationIndex"], countData["shiftStart"], 
                    countData["shiftEnd"], getStationResults=False,
                    getPairResults=True)
            
        dtype = {"names":["count", "mean", "variance"], 
                 'formats':[int, 'double', 'double']}
        result = np.zeros(self.pairCountData.shape, dtype=dtype)
        result["count"] = self.pairCountData
        result["mean"] = predictions["mean"]
        result["variance"] = predictions["variance"]
        return result
    
    
    def check_count_distributions_NB(self, minDataSetSize=20, fileName=None):
        
        self.prst("Checking whether the count data follow a negative "
                  + "binomial distribution")
        
        self.increase_print_level()
        
        allObservations = []
        allParams = []
        allCountData = self.shiftData[self.shiftData["prunedCount"] >= 0]
        totalCount = 0
        for stationIndex in self.usedStationIndexToStationIndex: 
            stationID = self.roadNetwork.stationIndexToStationID[stationIndex]
            self.prst("Station:", stationID)
            
            countData = allCountData[allCountData["stationIndex"] 
                                     == stationIndex]["prunedCountData"]
            
            if len(countData) < minDataSetSize:
                continue
            
            observations = FlexibleArrayDict(
                (3*max(len(countData[0]), 100), 
                 countData.size)
                )
            params = FlexibleArrayDict(
                (3*max(len(countData[0]), 100), 2)
                )
            
            for i, obsdict in enumerate(countData):
                for pair, count in obsdict.items():
                    if pair not in params.indexDict:
                        k, q = self._get_k_q(pair, 
                                 self.pruneStartTime, self.pruneEndTime, 
                                 stationIndex)
                        params.add(pair, [k, 1-q])
                    observations.get(pair)[i] = count
            """
            a = observations.get_array()
            for i in range(a.shape[0]):
                m, v = np.mean(a[i]), np.var(a[i], ddof=1)
                print(m, v)
            #"""
            allObservations.append(observations.get_array())
            allParams.append(params.get_array())
            totalCount += observations.size
            
        pModelAgg = []
        self.prst("Computing p-values ({} distributions)".format(totalCount))
        resolution = 50
        testN = 20
        counter = Counter(totalCount*testN, 0.01)
        
        self.increase_print_level()
        for i in range(testN):
            pValuesModel = []
            modelData = []
            
            with ProcessPoolExecutor() as pool:
                modelPs = pool.map(anderson_darling_NB, iterchain(*allObservations),
                                   iterchain(*allParams), repeat(False), 
                                   repeat(100)) 
                for pModel, cModel in modelPs:
                    modelData.append(cModel)
                    pValuesModel.append(pModel)
                    
                    percentage = counter.next()
                    if percentage: self.prst(percentage, percent=True)
                
            modelData = np.concatenate(modelData)
            ecmf = ECDF(modelData.ravel())
            x = np.unique(modelData)
            pModelAgg.append(anderson_darling_test_discrete(pValuesModel, x, ecmf(x), True))
        self.decrease_print_level()
        self.prst("p-value that data come from model", np.mean(pModelAgg))
        self.prst("Standard deviation:", np.std(pModelAgg), "| Results", pModelAgg)
        #sumv = np.array([np.sum(r) for r in iterchain(*allObservations)])
        #pValues = pValues[sumv>2,:]
        
        pmfhist = lambda x: np.histogram(x, int(np.max(x))+1, [0, int(np.max(x))+1])
        def reducedPmfhist(x):
            freq, bins = pmfhist(x)
            mask = freq > 0
            return freq[mask], bins[:-1][mask]
        
        lenv = np.array([len(r) for r in iterchain(*allObservations)]) #[sumv>2]
        
        maxv = [np.max(r) for r in iterchain(*allObservations)]
        self.prst("Distribution of the maximal count values", *reducedPmfhist(maxv))
        sumv = np.array([np.sum(r) for r in iterchain(*allObservations)])
        self.prst("Distribution of the total count values", *reducedPmfhist(sumv))
        self.prst("Distribution of sample sizes", *reducedPmfhist(lenv))
        
        
        
        c = Counter(totalCount*testN, 0.01)
        def counter():
            perc = c.next()
            if perc:
                self.prst(perc, percent=True)
        self.prst("Computing p-values to check the general distribution")
        pNBs = [anderson_darling_P(iterchain(*allObservations), False, 200, 200, 200, counter)
                for i in range(testN)]
        self.prst("p-value that data are NB distributed", np.mean(pNBs))
        self.prst("Standard deviation:", np.std(pNBs), "| Results", pNBs)
        c.reset()
        pPoiss = [anderson_darling_P(iterchain(*allObservations), True, 200, 200, 200, counter)
                for i in range(testN)]
        self.prst("p-value that data are Poisson distributed", np.mean(pPoiss))
        self.prst("Standard deviation:", np.std(pPoiss), "| Results", pPoiss)
        
        
        freqM, bins = np.histogram(pValuesModel, resolution, [0, 1])
        xModelRes, bins = np.histogram(modelData, resolution, [0, 1])
        
        if fileName is not None:
            self.prst("Plotting p-values")
            #plt.plot(np.insert(np.cumsum(xNBres), 0, 0), 
            #         np.insert(np.cumsum(freq)/np.sum(freq), 0, 0))
            #plt.plot(np.insert(np.cumsum(xPoisres), 0, 0), 
            #         np.insert(np.cumsum(freqP)/np.sum(freqP), 0, 0))
            #plt.plot(np.insert(np.cumsum(xModelRes)/np.sum(xModelRes), 0, 0), 
            plt.plot(np.insert(np.cumsum(xModelRes)/np.sum(xModelRes), 0, 0), 
                     np.insert(np.cumsum(freqM)/np.sum(freqM), 0, 0))
            #plt.plot(bins, np.insert(np.cumsum(freqP1)/np.sum(freqP1), 0, 0))
            #plt.plot(bins, np.insert(np.cumsum(freq10/np.sum(freq10)), 0, 0))
            #plt.plot(bins, np.insert(np.cumsum(freq20/np.sum(freq20)), 0, 0))
            plt.plot([0,1], [0,1], 'k--')
            plt.savefig(fileName + ".pdf")
            plt.savefig(fileName + ".png", dpi=1000)
            
            
        
        self.decrease_print_level()
        
    
    def get_PMF_observation_prediction(self, staionID, fromID, toID, xMax=None, 
                                       getBestPMF=True, 
                                       getPureObservations=False):
        
        stationIndex = self.roadNetwork.stationIDToStationIndex[staionID]
        fromIndex = self.roadNetwork.sourceIDToSourceIndex[fromID]
        toIndex = self.roadNetwork.sinkIDToSinkIndex[toID]
        
        countData = self.shiftData
        countData = countData[countData["stationIndex"] == stationIndex]
        countData = countData[countData["prunedCount"] >= 0]
        
        if not len(countData):
            raise ValueError("No observations have been made at the specified " +
                             "inspection spot. I stop comparing the distributions."
                             )
            
        pair = (fromIndex, toIndex)
        observations = [obsdict.get(pair, 0) 
                        for obsdict in countData["prunedCountData"]]
        
        observations = np.array(observations)
        
        if xMax is None:
            xMax = np.max(observations)
        else: 
            xMax = max(np.max(observations), xMax)
        
        observedPMF, _ = np.histogram(observations, xMax+1, (0, xMax+0.5))
        observedPMF = observedPMF / np.sum(observedPMF)
        
        
        X = np.arange(xMax+1, dtype=int)
                
        k, q = self._get_k_q((fromIndex, toIndex), 
                             self.pruneStartTime, self.pruneEndTime, 
                             stationIndex)
        
        if hasattr(k, "__iter__"): k = k[0]
        
        predictedPMF = nbinom.pmf(X, k, 1-q)
        
        result = (X, observedPMF, predictedPMF)
        
        if getBestPMF:
            def negLogLikelihood(params):
                kk, qq = params
                return -np.sum(nbinom.logpmf(observations, np.exp(kk),
                                             1-np.exp(-qq*qq)))
            
            x0 = [0, 1] 
            optResult = op.basinhopping(negLogLikelihood, x0, 40, 
                                        minimizer_kwargs={"method":"SLSQP",
                                                  "options":{"maxiter":300}})
            
            bestK, bestQ = optResult.x
            bestK = np.exp(bestK)
            bestQ = np.exp(-bestQ*bestQ)
            
            self.prst("Optimal k and q for this pair and this station",
                      "compared to the predicted k and q:")
            self.increase_print_level()
            self.prst("Optimal k:", bestK, "- Predicted k:", k)
            self.prst("Optimal q:", bestQ, "- Predicted q:", q)
            self.decrease_print_level()
            
            bestPMF = nbinom.pmf(X, bestK, 1-bestQ)
            
            result += (bestPMF, )
            
        if getPureObservations:
            result += (observations,)
        
        return result
        
    
    def compare_travel_time_distributions(self, saveFileName=None):
        self.prst("Creating traffic distribution comparison plots")
        self.increase_print_level()
        
        # create comparison plots
        times = np.linspace(0, 24, 5000)
        yLabel = "Traffic Density"
        xLabel = "Time"
        
        # compare long distance versus short distance
        plt.figure()
        longDistDistribution = self.__create_travel_time_distribution(None, True)
        restDistribution = self.__create_travel_time_distribution(None, False)
        H0Distribution = self.__create_travel_time_distribution(None, None,
                            {key:val for key, val 
                             in enumerate(iterchain((
                                 self.longDistTimeData.values()), 
                                 self.restTimeData.values()))}) #dict must be merged without overwriting keys!
        
        
        LR = 2 * (H0Distribution.negLL - longDistDistribution.negLL - 
                  restDistribution.negLL)
        
        p = chi2.sf(LR, df=2)
        
        plh = "   "
        
        self.prst("H0: Long distance and rest have same distribution.")
        self.prst(plh, "-2*ln(LR):", LR)
        self.prst(plh, "p-Value:", p)
        
        plt.plot(times, longDistDistribution.pdf(times), 
                 label="Long-Distance Traffic")
        plt.plot(times, restDistribution.pdf(times), label="Other Traffic")
        
        plt.ylabel(yLabel)
        plt.xlabel(xLabel)
        plt.legend()
        if saveFileName is not None:
            fn = saveFileName + "_LongVsRest"
            plt.savefig(fn + ".pdf")
            plt.savefig(fn + ".png")
        
        longDistTimeData = {key:val for key, val in 
                            self.longDistTimeData.items() if len(val)>=50}
        restTimeData = {key:val for key, val in 
                        self.restTimeData.items() if len(val)>=50}
        
        # compare traffic at the different sites
        for data, long, nameExt in ((longDistTimeData, True, "long"),
                                    (restTimeData, False, "rest")):
            if not len(data):
                continue
            
            LR = np.zeros((len(data), len(data)))
            nLL = np.zeros(len(data))
            
            plt.figure()
            keyList = list(data.keys())
            for i, label in enumerate(keyList):
                try:
                    dist = self.__create_travel_time_distribution(label, long)
                    nLL[i] = dist.negLL
                    if np.isfinite(dist.negLL):
                        plt.plot(times, dist.pdf(times), 
                             label=str(self.roadNetwork.stationIndexToStationID[
                                                                        label]))
                except RuntimeError:
                    pass
                for j in range(i+1, len(keyList)):
                    label2 = keyList[j]
                    try:
                        dist = self.__create_travel_time_distribution([label, 
                                                                   label2], 
                                                                  long)
                        LR[i,j] = LR[j,i] = dist.negLL
                    except RuntimeError:
                        pass
            
            nLLH0 = nLL + nLL[:,None]
            LR -= nLLH0
            LR *= 2
            
            # nan matrix entries result here only from inf-inf. This, in turn,
            # happens only if all entries are similar, which implies that
            # the MLs are equal
            #LR[np.isnan(nLLH0)] = 0
            
            # we cannot draw inference over distributions with too few data
            # therefore, we set those with too few data to nan.
            LR[np.isinf(nLLH0)] = np.nan
            np.fill_diagonal(LR, 0)
            p = chi2.sf(LR, df = 2)
            
            self.prst("Considered stations:", 
                      *self.roadNetwork.stationIndexToStationID[
                           list(data.keys())])
            self.prst("Respective numbers of data points:", 
                      *(len(val) for val in data.values()))
            
            self.prst("H0: Locations have pair-wise same distribution. (" 
                      + nameExt + ")")
            self.prst(plh, "-2*ln(LR):")
            print(np.round(LR,3))
            self.prst(plh, "p-Value:")
            print(np.round(p,3))
            
            someInfNan = not np.isfinite(nLL).all()
            if someInfNan:
                warnings.warn("Some likelihood values are NaNs or Infs. "+
                              "The following reuslt may be biased.")
                
            H0Distribution = self.__create_travel_time_distribution(None, None, 
                                                                data)
            LR = 2 * (H0Distribution.negLL - np.sum(nLL[np.isfinite(nLL)]))
            p = chi2.sf(LR, df = 2*np.sum(np.isfinite(nLL))-2)
            self.prst("H0: All locations have same distribution. (" 
                      + nameExt + ")")
            self.prst(plh, "-2*ln(LR):", LR)
            self.prst(plh, "p-Value:", p)
            
            
            plt.ylabel(yLabel)
            plt.xlabel(xLabel)
            plt.legend()
            if saveFileName is not None:
                fn = saveFileName + "_" + nameExt
                plt.savefig(fn + ".pdf")
                plt.savefig(fn + ".png")
            
        self.decrease_print_level()
        
        
    def compare_distributions(self, stationID, fromID, toID, xMax=None, 
                              saveFileName=None):
        try:
            X, observedPMF, predictedPMF, bestPMF, observations = \
                self.get_PMF_observation_prediction(stationID, fromID, toID, 
                                        xMax, getPureObservations=True)
        except KeyError:
            warnings.warn("One of the indices could not be resolved. " +
                          "I stop comparing the distributions.")
            return
        except ValueError as e:
            warnings.warn(str(e))
            return
    
        observedCMF = np.cumsum(observedPMF)
        predictedCMF = np.cumsum(predictedPMF)
        bestCMF = np.cumsum(bestPMF)
        
        self.prst("Testing whether the distribution for the flow from", 
                  fromID, "to", toID, "as observed at", stationID, "matches",
                  "the expectations.")
        
        p = anderson_darling_test_discrete(observations, X, predictedCMF, 
                                           True)
        
        self.increase_print_level()
        self.prst("p-value (for H0: the distributions are equal):", p)
        self.decrease_print_level()
        
        
        if saveFileName is not None:
            saveFileNamePMF = saveFileName + "PMF"
            saveFileNameCMF = saveFileName + "CMF"
        else:
            saveFileNamePMF = saveFileNameCMF = None
            
        create_distribution_plot(X, observedPMF, predictedPMF, bestPMF, "PMF", 
                                 fileName=saveFileNamePMF)
        create_distribution_plot(X, observedCMF, predictedCMF, bestCMF, "CMF", 
                                 fileName=saveFileNameCMF)
        
        return p
    
    def test_1_1_regression(self, minSampleSize=20, saveFileName=None,
                            comparisonFileName=None):
        
        self.prst("Performing a 1:1 regression test.")
        self.increase_print_level()
        
        if not os.access(saveFileName, os.F_OK): os.makedirs(saveFileName)
        saveFileName = os.path.join(saveFileName, saveFileName)
        
        if comparisonFileName:
            comparisonFileName = os.path.join(comparisonFileName, comparisonFileName)
        
        self.prst("Computing normalized observation and prediction values.")
        
        regressionData = self.get_normalized_observation_prediction(
                                                                minSampleSize)
        
        if regressionData.size <= 2:
            self.prst("Too few stations have the required sample size.",
                      "No regression analysis is possible.",
                      "Try to reduce minSampleSize.")
            self.decrease_print_level()
            return
        
        X, Y = regressionData["predicted"], regressionData["observed"]
        
        self.prst("Doing regression.")
        slope, intercept, r_value, p_value, std_err = linregress(X, Y)
        self.increase_print_level()
        
        self.prst("Slope:                         ", slope)
        self.prst("Intercept:                     ", intercept)
        self.prst("Correlation coefficient:       ", r_value)
        self.prst("R squared:                     ", R2(X, Y))
        self.prst("p-value for H0: no correlation:", p_value)
        self.prst("Standard error:                ", std_err)
                                        
        # compute the F-statistic
        n = regressionData.size
        RMSE = np.sum((Y - (intercept+slope*X))**2) / (n-2)
        F = (n*intercept**2 + 2*intercept*(slope-1)*np.sum(X)
             + (slope-1)**2 * np.sum(X**2)) / (2 * RMSE)
        self.prst("f-value:                     ", F)
        
        
        def fInt(p):
            a, b = fdist.interval(1-p, 2, n-2)
            return np.minimum(F-a, b-F)
        
        p = op.brenth(fInt, 0, 1)
        
        self.prst("p-value for H0: slope=1 and intercept=0:", p)
        
        for level in 98, 95, 90, 80, 70, 50, 25, 10, 5, 2:
            lbound, ubound = fdist.interval(level/100, 2, n-2)
            
            self.prst("Reject slope=1 and intercept=0 on a", level, 
                      "% significance level:", not lbound <= F <= ubound)
            
        self.decrease_print_level()
        self.prst("Creating regression plot.")
        
        if saveFileName is not None:
            saveFileName += "Regression"
            
        create_observed_predicted_mean_error_plot(X, Y, None, 1.96, None,
                                                  (slope, intercept),
                      title="Observed vs. Predicted Regression Analysis",
                      fileName=saveFileName,
                      comparisonFileName=comparisonFileName 
                      )
        create_observed_predicted_mean_error_plot(regressionData["predictedMeanScaled"], regressionData["observedMeanScaled"], None, 1.96,
                      title="Observed vs. Predicted Regression Analysis",
                      fileName=non_join(saveFileName, "_scaled"),
                      comparisonFileName=non_join(comparisonFileName, "_scaled")
                      )
        
        """
        modelData = self.routeModel
        p = 1-self._convert_parameters(modelData["parameters"], modelData["covariates"])[1]
        def errFuncUp(x):
            result = normaldist.ppf(0.975, x, np.sqrt(x/p))
            result[np.isnan(result)] = 0
            return result
        def errFuncLow(x):
            result = normaldist.ppf(0.025, x, np.sqrt(x/p))
            result[np.isnan(result)] = 0
            return result
        
        create_observed_predicted_mean_error_plot(X, regressionData["observedMean"], None, None,
                                                  (errFuncLow, errFuncUp),
                      title="Observed vs. Predicted Regression Analysis",
                      fileName=saveFileName + "Reg"
                      )
        """
        
        self.decrease_print_level()
    
    def create_quality_plots(self, worstLabelNo=5,
                             saveFileName=None,
                             comparisonFileName=None):
        
        """
        
        .. todo:: Compute mean at stations only, incorporate timing later.
            This could speed up the procedure significatnly.
        """
        
        #!!!!!!!!!!!!!!! porting an old version of the program.
        if not hasattr(self, "destinationData"):
            self.destinationData = self.lakeData
            del self.lakeData 
            self.originData = self.jurisdictionData
            del self.jurisdictionData
            self.rawDestinationData= self.rawLakeData
            del self.rawLakeData
            self.rawOriginData = self.rawJurisdictionData
            del self.rawJurisdictionData
        
        
        if not os.access(saveFileName, os.F_OK): os.makedirs(saveFileName)
        saveFileName = os.path.join(saveFileName, saveFileName)
        
        if comparisonFileName:
            comparisonFileName = os.path.join(comparisonFileName, comparisonFileName)
        
        self.prst("Creating quality plots.")
        
        countData = self.shiftData
        rawStationData, rawPairData = self.get_station_mean_variance(
                    countData["stationIndex"], countData["shiftStart"], 
                    countData["shiftEnd"], getStationResults=True,
                    getPairResults=True)
        stationData = self.get_station_observation_prediction(rawStationData)
        self.prst("Creating plot of the quality by station.")
        station_std = np.sqrt(stationData["variance"].ravel())
        create_observed_predicted_mean_error_plot(
            stationData["mean"].ravel(),
            stationData["count"].ravel(),
            station_std,
            None, None, None,
            stationData["stationID"].ravel(),
            "Predicted and observed boater flows by station",
            non_join(saveFileName, "Stations"),
            )
        create_observed_predicted_mean_error_plot(
            stationData["mean"].ravel(),
            stationData["count"].ravel(),
            fileName=non_join(saveFileName, "Stations_raw"),
            comparisonFileName=non_join(comparisonFileName, "Stations_raw")
            )
        create_observed_predicted_mean_error_plot(
            stationData["mean"].ravel()/station_std,
            stationData["count"].ravel()/station_std,
            fileName=non_join(saveFileName, "Stations_scaled"),
            comparisonFileName=non_join(comparisonFileName, "Stations_scaled")
            )
        
        self.prst("Creating plot of the quality by pair.")
        pairData = self.get_pair_observation_prediction(rawPairData)
        pair_std = np.sqrt(pairData["variance"].ravel())
        create_observed_predicted_mean_error_plot(
            pairData["mean"].ravel(),
            pairData["count"].ravel(),
            pair_std,
            title="Predicted and observed boater flows by source-sink pair",
            fileName=non_join(saveFileName, "Pairs")
            )
        create_observed_predicted_mean_error_plot(
            pairData["mean"].ravel(),
            pairData["count"].ravel(),
            fileName=non_join(saveFileName, "Pairs_raw"),
            comparisonFileName=non_join(comparisonFileName, "Pairs_raw")
            )
        create_observed_predicted_mean_error_plot(
            pairData["mean"].ravel()/pair_std,
            pairData["count"].ravel()/pair_std,
            fileName=non_join(saveFileName, "Pairs_scaled"),
            comparisonFileName=non_join(comparisonFileName, "Pairs_scaled")
            )
        
        self.prst("Creating plot of the quality by origin.")
        mean = np.sum(pairData["mean"], 0).ravel()
        count = np.sum(pairData["count"], 0).ravel()
        if worstLabelNo >= self.originData.size:
            labels = self.originData["destinationID"]
        else:
            diff = np.abs(mean-count)
            max10DiffInd = np.argpartition(diff, -worstLabelNo)[-worstLabelNo:]
            labels = np.empty_like(mean, dtype=object)
            labels[max10DiffInd] = self.originData["destinationID"][max10DiffInd]
        destination_std = np.sqrt(np.sum(pairData["variance"], 0)).ravel()
        create_observed_predicted_mean_error_plot(
            mean,
            count,
            destination_std,
            title="Predicted and observed boater flows by destination",
            labels=labels,
            fileName=non_join(saveFileName, "Destinations")
            )
        create_observed_predicted_mean_error_plot(
            mean, count,
            fileName=non_join(saveFileName, "Destinations_raw"),
            comparisonFileName=non_join(comparisonFileName, "Destinations_raw")
            )
        create_observed_predicted_mean_error_plot(
            mean/destination_std, count/destination_std,
            fileName=non_join(saveFileName, "Destinations_scaled"),
            comparisonFileName=non_join(comparisonFileName, "Destinations_scaled")
            )
        
        self.prst("Creating plot of the quality by origin.")
        mean = np.sum(pairData["mean"], 1).ravel()
        count = np.sum(pairData["count"], 1).ravel()
        if worstLabelNo >= self.destinationData.size:
            labels = self.destinationData["originID"]
        else:
            diff = np.abs(mean-count)
            max10DiffInd = np.argpartition(diff, -worstLabelNo)[-worstLabelNo:]
            labels = np.empty_like(mean, dtype=object)
            labels[max10DiffInd] = self.destinationData["originID"][
                                                                max10DiffInd]
        jur_std = np.sqrt(np.sum(pairData["variance"], 1)).ravel()
        create_observed_predicted_mean_error_plot(
            mean,
            count,
            jur_std,
            title="Predicted and observed boater flows by origin",
            labels=labels,
            fileName=non_join(saveFileName, "Origins")
            )
        create_observed_predicted_mean_error_plot(
            mean, count,
            fileName=non_join(saveFileName, "Origins_raw"),
            comparisonFileName=non_join(comparisonFileName, "Origins_raw")
            )
        create_observed_predicted_mean_error_plot(
            mean/jur_std, count/jur_std,
            fileName=non_join(saveFileName, "Origins_scaled"),
            comparisonFileName=non_join(comparisonFileName, "Origins_scaled")
            )
        
        """
        self.prst("Creating plot of the quality by pair (log scale).")
        create_observed_predicted_mean_error_plot(
            pairData["mean"].ravel(),
            pairData["count"].ravel(),
            np.sqrt(pairData["variance"].ravel()),
            title="Predicted and observed boater flows by source-sink pair",
            fileName=non_join(saveFileName, "Pairs_logScale"),
            logScale=True
            )
        create_observed_predicted_mean_error_plot(
            pairData["mean"].ravel(), pairData["count"].ravel(),
            fileName=non_join(saveFileName, "Pairs_logScale_raw"),
            logScale=True,
            comparisonFileName=non_join(comparisonFileName, "Pairs_logScale_raw")
            )
        """
        
        plt.show()
    
    def save_model_predictions(self, fileName=None):
        if fileName is None:
            fileName = self.fileName
        if fileName is None:
            raise ValueError("The model must have an internal file name or"
                             + "fileName must be explicitely given.")
        
        self.prst("Saving the model predictions to file", fileName)
        stationData = self.get_station_mean_variance(fullCompliance=True,
                                                     correctData=True)
        df = pd.DataFrame(stationData)
        df.to_csv(fileName + "Stations.csv", index=False)
        
        measures = [nbinom.mean, nbinom.var, nbinom.median, nbinom.ppf, 
                    nbinom.ppf, nbinom.ppf, nbinom.ppf]
        args = [None, None, None, 0.75, 0.9, 0.95, 0.975]
        pairData = self.get_pair_distribution_measure(measures, args)
        dtype = {"names":["fromID", "toID", "mean", "variance", "median", 
                          "75_percentile", "90_percentile",
                          "95_percentile", "97.5_percentile"], 
                 'formats':[IDTYPE, IDTYPE, 'double', 'double', 'double', 
                            'double', 'double', 'double', 'double']}
        result = np.zeros(pairData[0].size, dtype=dtype)
        
        for data, name in zip(pairData, dtype["names"][2:]):
            result[name] = data.ravel()
            
        fromID = self.roadNetwork.vertices.array["ID"][
                                      self.roadNetwork.sourceIndexToVertexIndex]
        toID = self.roadNetwork.vertices.array["ID"][
                                      self.roadNetwork.sinkIndexToVertexIndex]
        result["fromID"] = np.repeat(fromID, len(toID))
        result["toID"] = np.tile(toID, len(fromID))
        
        df = pd.DataFrame(result)
        df.to_csv(fileName + "Pair.csv", index=False)
        
        result = result.reshape((len(fromID), len(toID)))
        originResult = np.zeros(len(fromID), 
                                      dtype=[("fromID", IDTYPE),
                                             ("mean", 'double'),
                                             ("variance", 'double'),
                                             ("95_percentile", 'double'),
                                             ])
        originResult = np.zeros(len(toID), 
                                      dtype=[("toID", IDTYPE),
                                             ("mean", 'double'),
                                             ("variance", 'double'),
                                             ("meanInfested", 'double'),
                                             ("varianceInfested", 'double'),
                                             ("95_percentile", 'double'),
                                             ("95_percentileInfested", 'double'),
                                             ])
        alpha = 0.95
        def get_ppf(mean, var):
            p = mean / var
            n = mean**2/(var-mean)
            return nbinom.ppf(alpha, n, p)
        
        originResult["fromID"] = result["fromID"][:,0]
        originResult["mean"] = result["mean"].sum(1)        
        originResult["variance"] = result["variance"].sum(1)  
        originResult["95_percentile"] = get_ppf(
            originResult["mean"], originResult["variance"])
        
        assert (originResult["fromID"]==self.destinationData["originID"]).all()
        
        originResult["toID"] = result["toID"][0]
        originResult["mean"] = result["mean"].sum(0)        
        originResult["variance"] = result["variance"].sum(0)        
        originResult["meanInfested"] = result["mean"][self.destinationData["infested"]].sum(0)        
        originResult["varianceInfested"] = result["variance"][self.destinationData["infested"]].sum(0)        
        originResult["95_percentile"] = get_ppf(originResult["mean"], 
                                              originResult["variance"])
        originResult["95_percentileInfested"] = get_ppf(
            originResult["meanInfested"], originResult["varianceInfested"])
        
        
        df = pd.DataFrame(originResult)
        df.to_csv(fileName + "Origin.csv", index=False)
        df = pd.DataFrame(originResult)
        df.to_csv(fileName + "Destination.csv", index=False)
        
    
    # creates boater choices for one day. For each boater it returns
    # start, end, and path
    def simulate_count_data(self, stationTimes, day, parameters, covariates,
                            limitToOneObservation=False):
        
        routeModelData = self.routeModel
        routeLengths = routeModelData["routeLengths"].data
        
        params = self._convert_parameters(parameters, covariates)
        
        pRandom, routeExp, pObserve = routeModelData["routeChoiceModel"].parameters
        kMatrix = self._get_k_value(params, covariates) 
        
        q = params[1]
        
        
        # number of people going for each pair
        n1 = np.random.negative_binomial(kMatrix, 1-q)
        
        routePowers = sparsepower(routeLengths, routeExp)
        normConstants = sparsesum(routePowers)
        
        pairToPairIndex = self.roadNetwork._pair_to_pair_index
        
        multObservations = defaultdict(lambda: 0)
        
        stations = list(stationTimes.keys())
        times = list(stationTimes.values())
        shiftNo = len(stations)
        inspectedRoutes = routeModelData["inspectedRoutes"]
        observationsDType = [
            ("stationID", IDTYPE),
            ("day", int),
            ("shiftStart", "double"),
            ("shiftEnd", "double"),
            ("time", "double"),
            ("sourceID", IDTYPE),
            ("sinkID", IDTYPE),
            ]
        observations = FlexibleArray(10000, dtype=observationsDType)
        
        stationIndexToStationID = self.roadNetwork.stationIndexToStationID
        sourceIndexToSourceID = self.roadNetwork.sourceIndexToSourceID
        sinkIndexToSinkID = self.roadNetwork.sinkIndexToSinkID
        
        # boaters' route choices
        for source, row in enumerate(n1):
            for sink, n_ij in enumerate(row):
                pair = pairToPairIndex((source, sink))
                rp = routePowers[pair].data / normConstants[pair]
                choices = np.arange(rp.size)
                for _ in range(n_ij):
                    obsNum = 0
                    goRandom = np.random.rand() < pRandom
                    if goRandom:
                        observed = np.random.rand(shiftNo) < pObserve
                    else:
                        observed = np.zeros(shiftNo, dtype=bool)
                        route = np.random.choice(choices, p=rp)
                        for locInd, station in enumerate(stations):
                            observed[locInd] = (
                                station in inspectedRoutes and
                                (source, sink) in inspectedRoutes[station] and
                                route in inspectedRoutes[station][
                                    (source, sink)])
                    
                    tuples = []
                    for locInd in np.nonzero(observed)[0]:
                        time = self.travelTimeModel.sample()
                        start, end = times[locInd]
                        if start <= time <= end and np.random.rand() < self.complianceRate:
                            tuples.append((
                                stationIndexToStationID[stations[locInd]],
                                day,
                                start,
                                end,
                                time,
                                sourceIndexToSourceID[source],
                                sinkIndexToSinkID[sink]
                                ))
                    if len(tuples) == 1 or not limitToOneObservation or not goRandom:
                        for t in tuples:
                            observations.add_tuple(t)
                            obsNum += 1
                        if len(tuples):
                            multObservations[len(tuples)] += 1
        for s, t in zip(stations, times):
            start, end = t
            observations.add_tuple((
                stationIndexToStationID[s],
                day,
                start,
                end,
                0.,
                b'',
                b''
                ))
        
        observations = observations.get_array()
        observations.sort(order="stationID")
        
        return observations, multObservations
    
    
    
    
    def save_simulated_observations(self, parameters=None, covariates=None,
                                    shiftNumber=None,
                                    dayNumber = None,
                                    stationSets = None,
                                    fileName=None):
        
        if fileName is None: 
            fileName = self.fileName
            
        if parameters is None:
            parameters = self.routeModel["parameters"]
            covariates = self.routeModel["covariates"]
        
        self.prst("Simulating observations for statistic parameters", 
                  parameters, "with covariates", covariates)    
        
        print(self.travelTimeModel.location, self.travelTimeModel.kappa)
        
        if not shiftNumber:
            shiftData = self.shiftData.copy()
        else:
            shiftData = np.zeros(shiftNumber, dtype=self.shiftData.dtype)
            shiftData["shiftStart"] = np.maximum(np.minimum(
                                        np.random.vonmises(
                                            (-3)*np.pi/12, 5, shiftNumber)
                                                 *(12/np.pi) + 12, 23), 4)
            shiftData["shiftEnd"] = np.maximum(np.minimum(
                                        np.random.vonmises(
                                            3*np.pi/12, 5, shiftNumber)
                                                 *(12/np.pi) + 12, 23), 4)
            considered = shiftData["shiftStart"] >= shiftData["shiftEnd"]
            shiftData["shiftEnd"][considered] += shiftData["shiftStart"][
                                                                considered]+3
            
            shiftData["shiftEnd"] = np.minimum(shiftData["shiftEnd"], 23.5)
            
            pNewDay = dayNumber / len(shiftData)
            if stationSets is None:
                shiftData["stationIndex"] = np.random.choice(np.unique(
                                                    self.shiftData["stationIndex"]), 
                                                                 shiftNumber)
                dayIndex = 0
                for row in range(shiftData):
                    row["dayIndex"] = dayIndex
                    dayIndex += np.random.rand() < pNewDay
            else:
                dayIndex = 0
                shiftIndex = 0 
                stationIDToStationIndex = self.roadNetwork.stationIDToStationIndex
                while shiftIndex < len(shiftData):
                    chosenSet = np.random.choice(stationSets)
                    chosenStations = chosenSet[np.random.rand(len(chosenSet))
                                                              < 1/(pNewDay*len(chosenSet))]
                    for station in chosenStations:
                        if shiftIndex < len(shiftData):
                            shiftData["stationIndex"][shiftIndex] = \
                                stationIDToStationIndex[station]
                            shiftData["dayIndex"][shiftIndex] = dayIndex
                            shiftIndex += 1
                    dayIndex += 1
        
        #shiftData["shiftStart"]=11
        #shiftData["shiftEnd"]=16
        shiftData["shiftEnd"] += np.random.rand(shiftData.size)*1e-5
        
        observations = []
        countData = defaultdict(lambda: 0)
        
        newDay = ~(np.roll(shiftData["dayIndex"], -1) == shiftData["dayIndex"])
        newDay[-1] = True
        
        dayStartIndex = 0
        size = 0
        for dayEndIndex in np.nonzero(newDay)[0]+1:
            stationTimes = {}
            for shiftIndex in range(dayStartIndex, dayEndIndex):
                stationIndex = shiftData["stationIndex"][shiftIndex]
                stationTimes[stationIndex] = (shiftData["shiftStart"][shiftIndex],
                                              shiftData["shiftEnd"][shiftIndex])
            obsData, cData = self.simulate_count_data(
                stationTimes, shiftData["dayIndex"][dayStartIndex], parameters, 
                covariates)
            observations.append(obsData)
            size += obsData.size
            for obsNo, count in cData.items():
                countData[obsNo] += count
            dayStartIndex = dayEndIndex
        
        totalObsNo = 0
        distinctObsNo = 0
        for obsNo, count in countData.items():
            totalObsNo += obsNo*count
            distinctObsNo += count
            self.prst(count, "boaters have been observed", obsNo, "time(s).")
        self.prst("In total, we have", totalObsNo, "observations of",
                  distinctObsNo, "distinct boaters in", shiftData.size,
                  "shifts.")
        observationsArr = np.zeros(size, dtype=observations[0].dtype)
        
        sectionStart = 0
        for subArr in observations:
            observationsArr[sectionStart:sectionStart+subArr.size] = subArr
            sectionStart += subArr.size
        
        df = pd.DataFrame(observationsArr)
        for name in "stationID", "sourceID":
            df[name] = df[name].apply(lambda x: str(x)[2:-1])
        df["sinkID"] = df["sinkID"].apply(lambda x: str(x)[3:-1])
        df.to_csv(fileName + "_SimulatedObservations.csv", index=False)
        
        self.prst("Done.")
    
    @staticmethod
    @add_doc(read_origin_data, read_destination_data, read_survey_data,
             read_postal_code_area_data, set_compliance_rate, TransportNetwork)
    def new(fileNameBackup,
            trafficFactorModel_class=None,
            fileNameEdges=None, 
            fileNameVertices=None,
            fileNameOrigins=None,
            fileNameDestinations=None,
            fileNamePostalCodeAreas=None,
            fileNameObservations=None,
            complianceRate=None,
            preprocessingArgs=None,
            edgeLengthRandomization=0.001,
            routeParameters=None,
            considerInfested=None,
            destinationToDestination=False,
            restart=False, 
            **restartArgs):
        """Constructs a new :py:class:`HybridVectorModel`, thereby reusing saved previous results if possible.
        
        Parameters
        ----------
        fileNameBackup : str
            Name of the file to load and save the model; without file extension
        trafficFactorModel_class : class
            Class representing the gravity model; must be inherited from 
            :py:class:`BaseTrafficFactorModel`
        considerInfested : bool 
            If given, only origins with the provided infestation state will be 
            considered to fit the model; see :py:meth:`HybridVectorModel.set_origins_considered`
        restart : bool 
            If ``True``, earlier results will be ignored and the model 
            will be constructed from scratch. 
        routeParameters : tuple
            Parameters defining which routes are deemed admissible. See
            :py:meth:`find_potential_routes`.
        ---------------------------------
        **restartArgs : keyword arguments
            The arguments below specify which parts of the model 
            construction process shall be repeated even if earler results are 
            available. If the arguments are set to ``True``, the respective
            model constrution process will take place (provided the necessary
            arguments, such as file names, are provided. 
        readOriginData : bools
            Read csv with data on boater origins
        readDestinationData : bool
            Read csv with data on boater destinations
        readPostalCodeAreaData : bool
            Read csv with population data for postal code regions
        readRoadNetwork : bool 
            Read csv with road network data
        findShortestDistances : boold
            Find the shortest distances between boater
            origins and destinations and between destinations and postal code 
            area centres
        readSurveyData : bool
            Read csv file with boater survey data
        properDataRate : float
            Rate of complete data (inferred from the data if not given)
        fitTravelTimeModel : bool
            Fit the model for boaters' travel timing
        travelTimeParameters : float[]
            If the traffic time model shall not be fitted but rather created 
            with known parameters, this argument contains these parameters
        extrapolateCountData : bool
            Prepare the boater observation data for the fit of the full 
            traffic flow model
        findPotentialRoutes : bool
            Determine potential routes that boaters might take
        fitRouteChoiceModel : bool 
            Fit the model assigning probabilities to the potential boater routes 
        routeChoiceParameters : float[] 
            If the route choice model shall not be fitted but rather created 
            with known parameters, this argument contains these parameters
        continueRouteChoiceOptimization : bool 
            If ``True``, the :py:obj:`routeChoiceParameters` will be interpreted 
            as initial guess rather than the best-fit parameters for the route 
            choice model
        preapareTrafficFactorModel : bool
            Prepare the traffic factor model
        fitFlowModel : bool
            Fit the gravity model for the vector flow between origins and 
            destinations
        permutations : bool[][]
            Permutations of parameters to be considered. The number of 
            columns must match the maximal number of parameters of the model 
            (see :py:attr:`BaseTrafficFactorModel.SIZE` and 
            :py:attr:`BaseTrafficFactorModel.PERMUTATIONS`)
        flowParameters : float[]
            If the flow model shall not be fitted but rather created with known 
            parameters, this argument contains these parameters
        continueTrafficFactorOptimization : bool 
            If ``True``, the :py:obj:`flowParameters` will be interpreted 
            as initial guess rather than the best fitting parameters for the 
            flow model
            
        """
        
        if restart or not exists(fileNameBackup + ".vmm"):
            model = HybridVectorModel(fileNameBackup, 
                                            trafficFactorModel_class, 
                                            destinationToDestination)
        else:
            print("Loading model from file", fileNameBackup + ".vmm")
            model = saveobject.load_object(fileNameBackup + ".vmm")
            model.fileName = fileNameBackup
            if "destinationToDestination" not in model.__dict__:
                model.destinationToDestination = False
        
        attrDict = model.__dict__
        restartArgs = defaultdict(lambda: False, restartArgs)
        
        if not destinationToDestination:
            if ((not "rawOriginData" in attrDict 
                 or restartArgs["readOriginData"])
                    and fileNameOrigins is not None):
                model.read_origin_data(fileNameOrigins)
                #model.save()
        
        if ((not "rawDestinationData" in attrDict 
             or restartArgs["readDestinationData"])
                and fileNameDestinations is not None):
            model.read_destination_data(fileNameDestinations)
            #model.save()
        
        if ((not "postalCodeAreaData" in attrDict 
             or restartArgs["readPostalCodeAreaData"])
                and fileNamePostalCodeAreas is not None):
            model.read_postal_code_area_data(fileNamePostalCodeAreas)
            #model.save()
        
        if ((not "roadNetwork" in attrDict 
             or restartArgs["readRoadNetwork"])
                and fileNameEdges is not None
                and fileNameVertices is not None):
            model.create_road_network(fileNameEdges, fileNameVertices, 
                                      preprocessingArgs, 
                                      edgeLengthRandomization)
            #model.save()
        
        if (not "complianceRate" in attrDict 
            or (complianceRate is not None 
                and model.complianceRate != complianceRate)):
            model.set_compliance_rate(complianceRate)
            model.prst("Compliance rate set to ", complianceRate)
            
        if considerInfested is not None:
            model.set_origins_considered(None, considerInfested)
            
        if ("roadNetwork" in attrDict and
            (not "shortestDistances" in model.roadNetwork.__dict__
             or restartArgs["findShortestDistances"])):
            model.find_shortest_distances()
        
        if not destinationToDestination: #!!!!!!!!!!!!!
            if ((not "shiftData" in attrDict 
                 or restartArgs["readSurveyData"])
                    and "roadNetwork" in attrDict
                    and fileNameObservations is not None):
                properDataRate = restartArgs.get("properDataRate", None)
                model.read_survey_data(fileNameObservations, 
                                            properDataRate=properDataRate)
                #model.save()
                
            if ((not "travelTimeModel" in attrDict 
                 or restartArgs["fitTravelTimeModel"])
                    and "longDistTimeData" in attrDict):
                travelTimeParameters = restartArgs.get("travelTimeParameters", None)
                model.create_travel_time_distribution(travelTimeParameters, model.fileName)
                #model.save()
        else:
            if restart: model.save()
        
        if ("roadNetwork" in attrDict and routeParameters is not None):
            
            if restartArgs["extrapolateCountData"]:
                model.__erase_extrapolate_data()
            
            if (model.__dict__.get("routeModel", None) is None or 
                    restartArgs["findPotentialRoutes"]):
                model.find_potential_routes(*routeParameters)
                model.save()
            
            if destinationToDestination:
                raise NotImplementedError("A model with destination to destination traffic"
                                          + " has not yet been"
                                          + " implemented completely.")
            
            if ("travelTimeModel" in attrDict):
                save = model.create_route_choice_model(restartArgs["createRouteChoiceModel"])
                save = model.preprocess_count_data() or save
                if save:
                    #model.save()
                    pass
            
                
            if ("travelTimeModel" in attrDict):
                
                save = False
                routeChoiceParameters = restartArgs.get("routeChoiceParameters", None)
                continueRouteChoiceOptimization = restartArgs[
                                            "continueRouteChoiceOptimization"]
                
                
                refit = restartArgs["fitRouteChoiceModel"]
                save = model.fit_route_model(refit, routeChoiceParameters,
                                             continueRouteChoiceOptimization)
                
                if save:
                    #model.save()
                    pass
                    
                flowParameters = restartArgs.get("flowParameters", None)
                continueTrafficFactorOptimization = restartArgs["continueTrafficFactorOptimization"]
                
                #if model.fit_flow_model(params, refit=refit, flowParameters=flowParameters):
                if ("trafficFactorModel" not in model.__dict__ or 
                        restartArgs["preapareTrafficFactorModel"]):
                    model.prepare_traffic_factor_model()
                    
                refit = restartArgs["fitFlowModel"]
                permutations = restartArgs.get("permutations", None)
                if model.fit_flow_model(permutations, refit, flowParameters, 
                                        continueTrafficFactorOptimization):
                    model.test_1_1_regression(20,
                                              model.fileName + str(routeParameters))
                    model.create_quality_plots(saveFileName=
                                               model.fileName+str(routeParameters))
                    save = True
                
                if save:
                    model.save()
            #model.create_quality_plots(params, 
            #                               model.fileName + str(params))
        return model    
            

def create_observed_predicted_mean_error_plot(predicted, observed, error=None,
                                              constError=None,
                                              errorFunctions=None,
                                              regressionResult=None,
                                              labels=None, 
                                              title="", fileName=None,
                                              comparisonFileName=None,
                                              logScale=False):
    plt.figure()
    plt.title(title)
    if logScale:
        addition = 0.1
        observed = observed + addition
        predicted = predicted + addition
        plt.yscale('log')
        plt.xscale('log')
    
    if error is None:
        error2 = 0
    else:
        error2 = error
        
    print(title, "R2:", R2(predicted, observed))
    xMax = np.max(predicted+error2)
    yMax = np.max(observed)
    
    if (comparisonFileName and os.access(comparisonFileName+"_pred.vmdat", os.R_OK) 
        and os.access(comparisonFileName+"_obs.vmdat", os.R_OK)):
        predicted2 = saveobject.load_object(comparisonFileName+"_pred.vmdat")
        observed2 = saveobject.load_object(comparisonFileName+"_obs.vmdat")
        comparison = True
        print(title, "R2 (comparison):", R2(predicted2, observed2))
        xMax = max(xMax, np.max(predicted2))
        yMax = max(yMax, np.max(observed2))
    else:
        comparison = False
        
    addFact = 1.15
    
    if regressionResult is not None:
        slope, intercept = regressionResult
        xMax *= addFact
        yMax *= addFact
        x = min((yMax-intercept) / slope, xMax)
        y = x * slope + intercept
        
        plt.plot((0, x), (intercept, y), color='C2', linestyle="-", 
                 linewidth=1)
        linestyle = "--"
    else:
        linestyle = "-"
    
    upperRight = min(yMax, xMax) * addFact
    plt.plot((0, upperRight), (0, upperRight), color='C1', linestyle=linestyle, 
             linewidth=0.8)
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    
    if constError is not None:
        plt.fill_between((0, upperRight), (constError, upperRight+constError),
                         (-constError, upperRight-constError), facecolor='red',
                         alpha=0.15)
    elif errorFunctions is not None:
        lowerError, upperError = errorFunctions
        xError = np.linspace(0, upperRight, 1000)
        plt.fill_between(xError, lowerError(xError),
                         upperError(xError), facecolor='red',
                         alpha=0.3)
    
    if error is None:
        plt.scatter(predicted, observed, marker='.')
        if comparison:
            plt.scatter(predicted2, observed2, marker='^', facecolors='none', 
                        edgecolors='g')
    else:
        plt.errorbar(predicted, observed, xerr=error, fmt='.', elinewidth=0.5, 
                     capsize=1.5, capthick=0.5)
    
    
    if labels is not None:
        for label, x, y in zip(labels, predicted, observed):
            if not label:
                continue
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', 
                                connectionstyle='arc3,rad=0'))
    
    if fileName is not None:
        saveobject.save_object(predicted, fileName+"_pred.vmdat")
        saveobject.save_object(observed, fileName+"_obs.vmdat")
        plt.savefig(fileName + ".png", dpi=1000)
        plt.savefig(fileName + ".pdf")
    
    
def create_distribution_plot(X, observed, predicted=None, best=None, 
                             yLabel="PMF", title="", fileName=None):
    
    plt.figure()
    plt.title(title)
    
    plotArr = [observed]
    labels = ["Observed"]
    
    if predicted is not None:
        plotArr.append(predicted)
        labels.append("Model Prediction")
    if best is not None:
        plotArr.append(best)
        labels.append("Best Fit")
    
    colors = ['C0', 'C1', 'C2']
    
    X = np.append(X, X.size)
    
    for Y, color in zip(plotArr, colors):
        yy = np.append(Y, 0)
        plt.fill_between(X, 0, yy, step='post', alpha=.3, color=color)
    
    for Y, color, label in zip(plotArr, colors, labels):
        yy = np.insert(Y, 0, Y[0])
        plt.step(X, yy, color=color, label=label)
    
    plt.ylabel(yLabel)
    plt.xlabel("x")
    plt.legend()
    
    if fileName is not None:
        plt.savefig(fileName + ".png", dpi=1000)
        plt.savefig(fileName + ".pdf")
    
def create_observed_predicted_mean_error_plot_from_files(fileName1, fileName2, 
                                        extension="", **kwargs):
    fileName1 = os.path.join(fileName1, fileName1+extension)
    fileName2 = os.path.join(fileName2, fileName2+extension)
    predicted = saveobject.load_object(fileName1+"_pred.vmdat")
    observed = saveobject.load_object(fileName1+"_obs.vmdat")
    create_observed_predicted_mean_error_plot(predicted, observed,
                                              fileName=fileName1, 
                                              comparisonFileName=fileName2, 
                                              **kwargs)
    #plt.show()

def create_predicted_observed_box_plot(observedData, median, quartile1, 
                                       quartile3, percentile5, percentile95, 
                                       mean, labels=None, title="", 
                                       fileName=None):
    
    fig, axes = plt.subplots(1, 1)
    
    if labels is None:
        labels = range(len(percentile5))
    
    stats = []
    for p5, p25, p50, p75, p95, m, label in zip(percentile5, quartile1, 
                                                median, quartile3, 
                                                percentile95, mean, labels):
        
        item = {}
        item["label"] = label
        item["mean"] = m
        item["whislo"] = p5
        item["q1"] = p25
        item["med"] = p50
        item["q3"] = p75
        item["whishi"] = p95
        item["fliers"] = [] # required if showfliers=True
        stats.append(item)
        
    
    lineprops = dict(color='purple')
    boxprops = dict(color='green')
    
    plt.boxplot(observedData)
    axes.bxp(stats, widths=0.5, boxprops=boxprops, whiskerprops=lineprops, 
             medianprops=lineprops)
    plt.title(title)
    
    if fileName is not None:
        plt.savefig(fileName + ".png", dpi=1000)
        plt.savefig(fileName + ".pdf")
        

    

def probability_equal_lengths_for_distinct_paths(edgeNumber=1000, 
                                                 upperBound=None,
                                                 resultPrecision=0.1,
                                                 machinePrecision=1e-15,
                                                 experiments=100000):
    if not upperBound:
        upperBound = np.round(resultPrecision/edgeNumber**2
                              /machinePrecision)
    variance = ((upperBound+1)**2 - 1) / 12 * np.sqrt(edgeNumber)
    mean = upperBound / 2 * edgeNumber
    
    prob1 = 0
    endI = edgeNumber//2-1
    for i in range(endI+1):
        print(i, prob1)
        end = i == endI
        x = np.arange(i*upperBound-0.5, (i+1)*upperBound+end)
        if not i:
            x[0] = 0
        elif end:
            x[-1] = mean
        cdf = normaldist.cdf(x, loc=mean, scale=np.sqrt(variance))
        prob = cdf
        prob[1:] = prob[1:]-prob[:-1]
        prob = prob[1:]
        prob *= prob
        prob1 += np.sum(prob)
    prob1 *= 2
    probAll = 1-np.power(1-prob1, experiments)
    return prob1, probAll

def nbinom_fit(data):
    f = lambda x: -np.sum(nbinom.logpmf(data, x[0], np.exp(-x[1]*x[1]))) 
    
    x0 = (1, 1)
    res = op.minimize(f, x0, method="SLSQP", options={"disp":True})
    
    
    return res.x[0], np.exp(-res.x[1]*res.x[1])

def nbinom_fit_test(n, k1, k2, p=0.4):
    
    data1 = nbinom.rvs(k1, p, size=n)
    data2 = nbinom.rvs(k2, p, size=n)
    data = np.concatenate((data1, data2))
    
    x = nbinom_fit(data)
    print(x)
    
    priorMean = nbinom.mean(k1, p) + nbinom.mean(k2, p)
    posteriorMean = nbinom.mean(*x)
    print(priorMean, 2*posteriorMean)

def draw_operating_hour_reward(kappa):
    
    
    times = np.linspace(0, 12, 10000)
    
    captured = 2*vonmises.cdf(12+times, kappa, 12, 12/np.pi) - 1
    
    plt.plot(2*times, captured)
    plt.xlabel("Operation time")
    plt.ylabel("Covered boater flow")
    
    plt.show()

def redraw_predicted_observed(fileName1, fileName2):
    
    for ext in "Regression", "Regression_scaled":
        print("Plotting", ext)
        create_observed_predicted_mean_error_plot_from_files(fileName1, 
                                                             fileName2,
                                                             ext,
                                                             constError=1.96)
    
    for ext in ["Stations_raw", "Stations_scaled", "Pairs_raw", "Pairs_scaled",
                "Destinations_raw", "Destinations_scaled", "Origins_raw",
                "Origins_scaled"]:
        print("Plotting", ext)
        create_observed_predicted_mean_error_plot_from_files(fileName1, 
                                                             fileName2, ext)
    
    plt.show()
    
def main():
    
    #nbinom_fit_test(1000, 1, 100, 0.5)
    #print(probability_equal_lengths_for_distinct_paths(1000, 1e5))
    #sys.exit()
    
    #draw_operating_hour_reward(1.05683124)
    #sys.exit()
    
    restart = True
    restart = False
    #print("test4")
     
      
    """ 
    
    stationSets = [
        np.array([b'6', b'9', b'20', b'6b']),
        np.array([b'14', b'18', b'22', b'6b', b'18b']),
        ]
    
    fileNameEdges = "LakeNetworkExample_full.csv"
    fileNameVertices = "LakeNetworkExample_full_vertices.csv"
    fileNameOrigins = "LakeNetworkExample_full_populationData.csv"
    fileNameDestinations = "LakeNetworkExample_full_lakeData.csv"
    fileNamePostalCodeAreas = "LakeNetworkExample_full_postal_code_areas.csv"
    fileNameObservations = "LakeNetworkExample_full_observations.csv"
    fileNameObservations = "LakeNetworkExample_full_observations_new2.csv"
    fileNameObservations = "shortExample3_SimulatedObservations.csv"
    fileNameObservations = "shortExample1_SimulatedObservations.csv"
    fileNameObservations = "LakeNetworkExample_full_observations_new.csv"
    fileNameComparison = ""
    complianceRate = 0.5
    
    fileNameSave = "shortExample3"
    fileNameSave = "shortExample2"
    fileNameSave = "shortExample1"
    fileNameSaveNull = "shortExampleNull"
    
    '''
    fileNameEdges = "LakeNetworkExample_mini.csv"
    fileNameVertices = "LakeNetworkExample_mini_vertices.csv"
    fileNameOrigins = "LakeNetworkExample_mini_populationData.csv"
    fileNameDestinations = "LakeNetworkExample_mini_lakeData.csv"
    fileNamePostalCodeAreas = "LakeNetworkExample_mini_postal_code_areas.csv"
    fileNameObservations = "LakeNetworkExample_mini_observations.csv"
    
    #fileNameEdges = "LakeNetworkExample_small.csv"
    #fileNameVertices = "LakeNetworkExample_small_vertices.csv"
    
    fileNameSave = "debugExample"
    #'''
    """ 
    fileNameEdges = "ExportEdges_HighwayTraffic.csv" 
    fileNameEdges = "ExportEdges.csv" 
    fileNameVertices = "ExportVertices.csv"
    fileNameOrigins = "ExportPopulation.csv"
    fileNameDestinations = "ExportLakes.csv"
    fileNamePostalCodeAreas = "ExportPostal_Code_Data.csv"
    fileNameObservations = "ZebraMusselSimulation_1.3-0.35-0.8-1.2-11000_SimulatedObservations.csv"
    fileNameObservations = "ZebraMusselSimulation_1.4-0.2-0.8-1.2-1_SimulatedObservations.csv"
    fileNameObservations = "ZebraMusselSimulation_1.4-0.2-0.8-1.2-1_SimulatedObservationsFalseAdded.csv"
    fileNameObservations = "ExportBoaterSurveyFalseRemoved.csv"
    fileNameObservations = "ZebraMusselSimulation_1.4-0.2-0.8-1.2-11000_SimulatedObservations.csv"
    fileNameObservations = "ZebraMusselSimulation_1.6-0.2-0.8-1.1-1_SimulatedObservations.csv"
    fileNameObservations = "ExportBoaterSurvey.csv"
    fileNameObservations = "ExportBoaterSurvey_HR_fit.csv"
    fileNameObservations = "ExportBoaterSurvey_HR_val.csv"
    fileNameObservations = "ExportBoaterSurvey_HR_with_days_fit.csv"
    fileNameObservations = "ExportBoaterSurvey_HR_with_days_validation.csv"
    complianceRate = 0.7959
    
    fileNameSave = "Sim_1.4-0.1-0.8-1.1-1_PathTest" # for road network only.
    fileNameSave = "Sim_1.4-0.2-1-1-1_HR_opt" # for optimization of inspection locations
    fileNameSave = "Sim_1.4-0.2-1-1-1_HR_HW_1" 
    fileNameSave = "Sim_1.4-0.2-1-1-1_HR_val" # for validation
    fileNameComparison = fileNameSave
    fileNameSave = "Sim_1.4-0.2-1-1-1_HR_HW" #for fit and road traffic estimates
    fileNameSaveNull = "Sim_1.4-0.2-1-1-1_HR_HW_null" #for fit and road traffic estimates
    fileNameSaveNull = "Sim_1.4-0.2-1-1-1_HR_val_null" #for fit and road traffic estimates
    #"""  
     
    #redraw_predicted_observed(fileNameSave, fileNameComparison)
    #sys.exit()
    
    print("Starting test. (2)")
     
    #print("Seed")
    
    #np.random.seed() 
    
    routeParameters = ( 
                          (1.4, .2, 1, 1)
                       )
    #"""
    #"""
    flowParameters = {}
    #best params one more parameter 
    flowParameters["covariates"] = np.array( 
        [True,True,True,False,False,False,True,False,True,False,True,False,True,True,True,False,False,False,True,False,False,False,True,True]
        )
    flowParameters["paramters"] = np.array(
        [-1.71042747e+01,1.15230704e+00,1.23546394e+03,5.55260234e+00,3.50775439e+00,2.53985567e+01,1.01026970e+03,8.86681452e+02,0,-1.8065007296786513,2.69364013e+00,-3.44611446e+00]
        )
    nullParameters = {}
    #best params one more parameter 
    nullParameters["covariates"] = flowParameters["covariates"].copy()
    nullParameters["covariates"][:] = False
    #nullParameters["covariates"][:3] = True
    nullParameters["paramters"] = np.array([-50., 10.])
    nullParameters["paramters"] = np.array([7.42643338e-01, 5.15536529e+04])
    
    #print(TrafficFactorModel.convert_parameters(None, flowParameters["paramters"][2:], flowParameters["covariates"][2:]))
    #sys.exit()
    """
    #best params one less parameter 
    flowParameters["covariates"] = np.array( 
        [True,True,True,False,False,False,True,False,True,False,True,False,True,False,False,False,False,False,True,False,False,False,True,True]
        )
    flowParameters["paramters"] = np.array(
        [-1.71042747e+01,1.15230704e+00,1.23546394e+03,5.55260234e+00,3.50775439e+00,2.53985567e+01,1.171835466180462,-1.8065007296786513,2.69364013e+00,-3.44611446e+00]
        )
    
    #best params
    flowParameters["covariates"] = np.array( 
        [True,True,True,False,False,False,True,False,True,False,True,False,True,True,False,False,False,False,True,False,False,False,True,True]
        )
    flowParameters["paramters"] = np.array(
        [-1.71042747e+01,1.15230704e+00,1.23546394e+03,5.55260234e+00,3.50775439e+00,2.53985567e+01,1.01026970e+03,8.86681452e+02,-1.8065007296786513,2.69364013e+00,-3.44611446e+00]
        )
        
    
    
    # best params old parameterization
    flowParameters["covariates"] = np.array( 
        [True,True,True,False,False,False,True,False,True,False,True,False,True,True,False,False,False,False,True,False,False,False,True,True]
        )
    flowParameters["paramters"] = np.array(
        [-1.71042747e+01,1.15230704e+00,1.23646394e+03,6.55260234e+00,4.50775439e+00,2.63985567e+01,1.01126970e+03,8.87681452e+02,1.64227810e-01,2.69364013e+00,-3.44611446e+00]
        )
    """
    routeChoiceParameters = [0.048790208690779414, -7.661288616999463, 0.0573827962901976]
    nullRouteChoiceParameters = [1, 0, 0.0001]
    travelTimeParameters = np.array([14.00344885,  1.33680321])
    nullTravelTimeParameters = np.array([0, 1e-10])
    properDataRate = 0.9300919842312746
    
    
    nullModel = HybridVectorModel.new(
                fileNameBackup=fileNameSaveNull, 
                trafficFactorModel_class=TrafficFactorModel,
                fileNameEdges=fileNameEdges,
                fileNameVertices=fileNameVertices,
                fileNameOrigins=fileNameOrigins,
                fileNameDestinations=fileNameDestinations,
                fileNamePostalCodeAreas=fileNamePostalCodeAreas,
                fileNameObservations=fileNameObservations,
                complianceRate=complianceRate,
                preprocessingArgs=None,
                #preprocessingArgs=(10,10,10),
                #considerInfested=True, 
                #findPotentialRoutes=True,
                edgeLengthRandomization=0.001,
                routeParameters=(1.0001, 0.9999, 0.5, 2), 
                readSurveyData=True,
                properDataRate=properDataRate,
                #createRouteChoiceModel=True,
                fitRouteChoiceModel=True,
                #readOriginData=True,
                #readOriginData=True, 
                travelTimeParameters=nullTravelTimeParameters, 
                fitTravelTimeModel=True,
                fitFlowModel=True,
                routeChoiceParameters=nullRouteChoiceParameters, #continueRouteChoiceOptimization=True,
                flowParameters=nullParameters, #continueTrafficFactorOptimization=True, #readDestinationData=True,  readPostalCodeAreaData=True, , #  #findPotentialRoutes=True, #  extrapolateCountData=True , # #readSurveyData=True   ###  #  #   findPotentialRoutes=False ,  readSurveyData=True 
                restart=restart, #readSurveyData=True, 
                )
    sys.exit()
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
                preprocessingArgs=None,
                #preprocessingArgs=(10,10,10),
                #considerInfested=True, 
                #findPotentialRoutes=True,
                edgeLengthRandomization=0.001,
                routeParameters=routeParameters, 
                #readSurveyData=True,
                properDataRate=properDataRate,
                #createRouteChoiceModel=True,
                #fitRouteChoiceModel=True,
                #readOriginData=True,
                #readOriginData=True, 
                fitFlowModel=True,
                routeChoiceParameters=routeChoiceParameters, continueRoutChoiceOptimization=False,
                flowParameters=flowParameters, continueTrafficFactorOptimization=False, #readDestinationData=True,  readPostalCodeAreaData=True, , #  #findPotentialRoutes=True, #  extrapolateCountData=True , # #readSurveyData=True   ###  #  #   findPotentialRoutes=False ,  readSurveyData=True 
                travelTimeParameters=travelTimeParameters, 
                #fitTravelTimeModel=True,
                restart=restart, #readSurveyData=True, 
                )
    #model.compare_travel_time_distributions(model.fileName)
    
    #sys.exit()
    
    '''
    model = saveobject.load_object(fileNameSave + ".vmm")
    #'''
    #"""
    #model.save_model_predictions()
    '''
    model.save_simulated_observations(shiftNumber=2500, dayNumber=600, 
                                      stationSets=stationSets)
    '''
    #"""
    model.create_quality_plots(saveFileName=model.fileName, worstLabelNo=5,
                               comparisonFileName=fileNameComparison)
    model.test_1_1_regression(20, saveFileName=model.fileName,
                              comparisonFileName=fileNameComparison)
    model.save(model.fileName)
    sys.exit()
    model.save_simulated_observations()
    model.check_count_distributions_NB(fileName=model.fileName+"-pvals")
    sys.exit()
    #"""
    #'''
    #model.optimize_inspection_station_placement(20, saveFileName=model.fileName)
    allowedShifts = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    allowedShifts = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    allowedShifts = [6, 9, 10, 14, 18]
    allowedShifts = [6., 7., 8., 9., 10., 11., 12., 13., 14., 16., 18., 20., 23.]
    allowedShifts = [4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22]
    allowedShifts = [2, 4 ]
    allowedShifts = [4, 6, 8, 9, 10, 11, 12, 13, 15, 17, 20, 22]
    allowedShifts = [5, 8, 10, 11, 12, 15, 19, 22]
    
    """
    RouteChoiceModel.NOISEBOUND = 0.2
    print("Setting noise bound to", RouteChoiceModel.NOISEBOUND)
    model.fit_route_model(routeParameters, True)
    RouteChoiceModel.NOISEBOUND = 0.02
    print("Setting noise bound to", RouteChoiceModel.NOISEBOUND)
    model.fit_route_model(routeParameters, True)
    """
    
    routeChoiceParamCandidates = [
                                  None,
                                  [0.00739813, -5.80269191, 0.7815986],
                                  [0.19972783, -5.69001292, 0.012684587],
                                  ]
    """
    RouteChoiceModel.NOISEBOUND = 0.2
    for r in routeChoiceParamCandidates:
        print("Setting Route Choice parameters to", r)
        model.fit_route_model(routeParameters, True, np.array(r))
        model.fit_flow_model(flowParameters=flowParameters, continueFlowOptimization=True,
                             refit=True)
    """
        
    def setRouteParameters(m, val, refit=True):
        if val is not None:
            m.NOISEBOUND = 1
            m.fit_route_model(routeParameters, True, np.array(val), False, get_CI=False)
            if refit:
                m.fit_flow_model(flowParameters=flowParameters, continueFlowOptimization=True,
                                 refit=True, get_CI=False)
    def setNoise(m, val):
        if val is not None:
            print("Set noise to", val)
            m.NOISEBOUND = 1
            routeParams = np.array(m.routeModel["routeChoiceModel"].parameters)
            routeParams[0] = val
            m.fit_route_model(True, routeParams, False, get_CI=False)
    defaultArgs = dict(costShift=3.5, costSite=1., shiftLength=8., costRoundCoeff=None, 
                       nightPremium=(5.5, 21, 5), baseTimeInv=24, timeout=3000,
                       init_greedy=True, costBound=80.)#, loadFile=False) 
    
    """
    model.set_infested(b'J54145')
    model.set_infested(b'J54170')
    model.set_infested(b'J54185')
    model.fileName += "Idaho"
    #"""
    model.save_model_predictions()
    
    #model.fileName += "_noinit_"
    #model.create_caracteristic_plot("costBound", [20., 80., 160.], **defaultArgs)
    #model.create_budget_plots(5, 75, 15, **defaultArgs)
    #model.create_budget_plots(55, 100, 10, **defaultArgs)
    #model.create_budget_plots(100, 150, 11, **defaultArgs)
    #model.create_budget_plots(135, 150, 4, **defaultArgs)
    
    sys.exit()
    #model.create_budget_plots(5, 150, 30, **defaultArgs)
    
    
    model.create_caracteristic_plot("costBound", [25., 50., 100.], characteristicName="Budget", **defaultArgs)
    
    #model.create_caracteristic_plot(setRouteParameters, routeChoiceParamCandidates, 
    #                                "NoiseRefit", [0.047, 0.007, 0.2],
    #                                **defaultArgs)
    noise = [0.001, 0.05, 0.2]
    #model.create_caracteristic_plot(setNoise, noise, "Noise level", **defaultArgs)
    
    sys.exit()
    
    for ignoreRandomFlow in False, True:
        #profile("model.optimize_inspection_station_operation(2, 1, 30, 6., allowedShifts=allowedShifts, costRoundCoeff=1, baseTimeInv=18, ignoreRandomFlow=ignoreRandomFlow, saveFileName=model.fileName)", 
        #        globals(), locals()) 
        #"""        
        model.optimize_inspection_station_operation(3.5, 1., 10., 7, #80
                                                    #allowedShifts=allowedShifts, #[6, 8, 10, 11, 12, 14], 
                                                    costRoundCoeff=0.5, 
                                                    nightPremium=(1.2, 22, 6),
                                                    baseTimeInv=24,
                                                    ignoreRandomFlow=ignoreRandomFlow,
                                                    integer=True,
                                                    extended_info=True
                                                    )
    #'''
    """
    print(find_shortest_path(model.roadNetwork.vertices.array,
                             model.roadNetwork.edges.array,
                             0, 9))
    """
    
    """
    stations = [b'386', b'307', b'28']
        
    fromIDs = [b'J54130', b'J54181', b'J54173']
    
    toIDs = [b'L329518145A', b'L328961702A', b'L328974235A']
    
    ps = []
    for stationID, fromID, toID in iterproduct(stations, fromIDs, toIDs):
        #print("Consider journeys from", fromID, "to", toID, "observed at", 
        #      stationID)
        fname = model.fileName + str(stationID + fromID + toID)
        p = model.compare_distributions(stationID, fromID, toID, 15, saveFileName=fname)
        if p is not None:
            ps.append(p)
    
    try:
        print("p distribution:", np.min(ps), np.max(ps), np.histogram(ps))
    except Exception:
        pass
       
    #"""
    #model.compare_distributions(b'3', b'1', b'L1', 15, saveFileName=model.fileName)
    #model.test_1_1_regression(20, routeParameters[0], model.fileName + str(routeParameters[0]))
    #model.save_simulated_observations(shiftNumber=1000, fileName=model.fileName+"1000")
    #model.save_simulated_observations()
    plt.show()
    
    """
    considered = np.array([True] * 7)
    #considered[2] = False
    #x0Stat = (np.log(5), np.sqrt(-np.log(0.7)), np.sqrt(-np.log(.1)), -3, np.sqrt(-np.log(0.1))) 
    #x0Flex = np.array((np.log(2), 1, np.log(1), -1, 1, 1, -2))
    x0Stat = (1, 0.5, 5, -3, 1.5) 
    x0Stat = (5.5, 0.5, 1.5, -3, 1.5) 
    x0Flex = np.array((2, 1, 1, -1, 1, 1, -2))
    x0 = np.array((*x0Stat, *x0Flex[considered]))
    #model.simulate_count_data_test(5, x0, covariates=considered)
    model.save_simulated_observations(x0, considered, "area", shiftNumber=2000, fileName=model.fileName+"1000")
    #"""
    
    """
    
    #profile("network = TransportNetwork(fileNameEdges, fileNameVertices)", globals(), locals())
    if exists(fileNameSave) and not restart:
        print("Loading file...")
        network = saveobject.load_object(fileNameSave)
        print("Edge number", len(network.edges))
        print("Edge size", network.edges.size)
    else:
        network = TransportNetwork(fileNameEdges, fileNameVertices)
        print("Edge number", len(network.edges))
        network.lock = None 
        print("Saving file...")
        saveobject.save_object(network, fileNameSave)
    
    #network.find_potential_routes(1.5, .2)
    #profile("network.find_potential_routes(1.5, .2)", globals(), locals())  
    #print("Timed execution:", Timer(network.find_potential_routes).timeit(1))
    #network.find_alternative_paths_test(1.5)
    for stretch, lo, prune in (
                               (1.25, .1, .7),
                               (1.5, .2, .7),
                               (1.25, .2, .7),
                               (1.25, .1, 1),
                               (1.25, .2, 1),
                               (1.5, .2, 1),
                               ):
        print("Timed execution:", Timer("network.find_potential_routes(stretch, lo, prune)", globals=locals()).timeit(1))
        print("="*80)
        print()
        print("="*80)
    """
    
if __name__ == '__main__':
    #f = "ZebraMusselSimulation_1.4-0.2-1-1-1_HR_opt[3.5, 1, 50.0, 8, (1.2, 22, 6), 0].dat"
    #o = saveobject.load_object(f)
    #print(o)
    
    
    main()
    # LD_PRELOAD=../anaconda3/lib/libmkl_core.so python ...