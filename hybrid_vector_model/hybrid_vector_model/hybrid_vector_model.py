'''

.. note:: This program was created with the motivation to model the traffic
    of boaters potentially carrying aquatic invasive species. Nonetheless,
    the tools are applicable to assess and control any vector road traffic. 
    Due to the initial motivation, however, the wording within this file may at
    some places still be specific to the motivating scenario:
            
    - We may refer to `vectors` as `boaters` or `agents`. 
    - We may refer to the origins of vectors as `origin`, `source`, or simply `jurisdiction`. 
    - We may refer to the destinations of vectors as `destination`, `sink`, 
      or `lake`.


'''

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
from scipy.stats import nbinom, f as fdist, linregress, chi2
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
from vemomoto_core.tools.doc_utils import DocMetaSuperclass, inherit_doc, staticmethod_inherit_doc
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

def _non_join(string1, string2):
    """Joins two strings if they are not ``None``.
    
    Returns ``None`` if either of the input strings is ``None``.
    
    Parameters
    ----------
    string1 : str 
        First string
    string2 : str 
        Second string
    
    """
    if string1 is not None and string2 is not None:
        return string1 + string2
    else:
        return None

def create_observed_predicted_mean_error_plot(predicted, observed, error=None,
                                              constError=None,
                                              errorFunctions=None,
                                              regressionResult=None,
                                              labels=None, 
                                              title="", saveFileName=None,
                                              comparisonFileName=None,
                                              logScale=False):
    """Create an observed vs. predicted plot.
    
    Parameters
    ----------
    predicted : float[]
        Array of predicted values.
    observed : float[]
        Array of observed values.
    errorFunctions : callable[]
        Two methods for the lower and the upper predicted error (e.g. 95% 
        confidence interval). Both of these methods must take the predicted
        value and return the repsective expected bound for the observations.
        If given, the area between these function will be plotted as a shaded 
        area.
    regressionResult : float[]
        Slope and intercept of an observed vs. prediction regression. If given,
        the regression line will be plotted.
    labels : str[]
        Labels for the data points.
    title : str
        Title of the plot
    saveFileName : str
        Name of the file where the plot and the data used to generate it will
        be saved.
        
        ~+~
        
        .. note:: Note that only predicted and observed values will be saved.
    comparisonFileName : str
        Name of the file where alternative results are saved. These results will
        be loaded and plotted for comparison.
    logScale : bool
        Whether to plot on a log-log scale.
    
    """
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
    
    if saveFileName is not None:
        saveobject.save_object(predicted, saveFileName+"_pred.vmdat")
        saveobject.save_object(observed, saveFileName+"_obs.vmdat")
        plt.savefig(saveFileName + ".png", dpi=1000)
        plt.savefig(saveFileName + ".pdf")
    
    
def create_distribution_plot(X, observed, predicted=None, best=None, 
                             yLabel="PMF", title="", fileName=None):
    """Creates a plot of a given discrete distribution.
    
    Parameters
    ----------
    X : float[]
        Values that could be observed.
    observed : float[]
        Observed cumulative density (non-parametric estimate of the cumulative 
        mass function).
    predicted : float[]
        Predicted cumulative density.
    best : float[]
        Second predicted cumulative density (different prediction method).
    yLabel : str
        Label of the y axis.
    title : str
        Title of the plot.
    fileName : str
        Name of the file that the plot shall be saved to.
    
    """
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
    """Creates an observed vs. predicted plot from saved data.
    
    Parameters
    ----------
    fileName1 : str
        Name of the file from which the primary data shall be loaded.
    fileName2 : str
        Name of the file from which comparison data shall be loaded.
    extension : str
        Extension to add to both file names.
    **kwargs : kwargs
        Keyword arguments passed to :py:meth:`create_observed_predicted_mean_error_plot`.
    
    """
    fileName1 = os.path.join(fileName1, fileName1+extension)
    fileName2 = os.path.join(fileName2, fileName2+extension)
    predicted = saveobject.load_object(fileName1+"_pred.vmdat")
    observed = saveobject.load_object(fileName1+"_obs.vmdat")
    create_observed_predicted_mean_error_plot(predicted, observed,
                                              saveFileName=fileName1, 
                                              comparisonFileName=fileName2, 
                                              **kwargs)

def nbinom_fit(data):
    """Fits a negative binomial distribution to given data"""
    f = lambda x: -np.sum(nbinom.logpmf(data, x[0], np.exp(-x[1]*x[1]))) 
    
    x0 = (1, 1)
    res = op.minimize(f, x0, method="SLSQP", options={"disp":True})
    
    
    return res.x[0], np.exp(-res.x[1]*res.x[1])

@inherit_doc(create_observed_predicted_mean_error_plot_from_files)
def redraw_predicted_observed(fileName1, fileName2):
    """Redraws predicted versus oberved plots generated earlier."""
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
        """Creates vertices that represent the destinations
        
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
        
        routeLengths, inspectedRoutes, stationCombinations = \
            FlowPointGraph.find_alternative_paths(self, 
                                                self.sourceIndexToVertexIndex, 
                                                self.sinkIndexToVertexIndex,
                                                self.shortestDistances, 
                                                stretchConstant, 
                                                localOptimalityConstant, 
                                                acceptionFactor,
                                                rejectionFactor)
        
        self.admissibilityParameters = (stretchConstant, localOptimalityConstant, 
                                        acceptionFactor, rejectionFactor)
        self.lengthsOfPotentialRoutes = routeLengths
        self.inspectedPotentialRoutes = inspectedRoutes
        self.stationCombinations = stationCombinations
        
    
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
    
    def convert_parameters(self, dynamicParameters, parametersConsidered):
        """Converts an array of given parameters to an array of standard (maximal)
        length and in the parameter domain of the model.
        
        Not all parameters may be parametersConsidered in the model (to avoid overfitting)
        Furthermore, some parameters must be constrained to be positive or 
        within a certain interval. In this method, the parameter vector 
        (containing only the values of the free parameters) is transformed to 
        a vector in the parameter space of the model
        
        Parameters
        ----------
        dynamicParameters : float[] 
            Free parameters. The parameters that are not held constant.
        parametersConsidered : bool[] 
            Which parameters are free? Is ``True`` at the entries corresponding 
            to the parameters that are free. ``parametersConsidered`` must have exactly as 
            many ``True`` entries as the length of ``dynamicParameters``
        
        """
        result = np.full(len(parametersConsidered), np.nan)
        result[parametersConsidered] = dynamicParameters
        return result
    
    def get_mean_factor(self, parameters, parametersConsidered, pair=None):
        """Returns a factor proportional to the mean traveller flow between the
        source-sink pair ``pair`` or all sources and sinks (if ``pair is None``)
        
        ~+~
        
        .. note:: This method MUST be overwritten. Otherwise the model will 
            raise an error.
        
        Parameters
        ----------
        parameters : double[] 
            Contains the free model parameters.
        parametersConsidered : bool[] 
            Which parameters are free? Is ``True`` at the entries corresponding 
            to the parameters that are free. ``parametersConsidered`` must have exactly as 
            many ``True`` entries as the length of ``dynamicParameters``.
        pair : (int, int) 
            Source-sink pair for which the factor shall be determined.
            This is the source-sink pair of interest (the indices of the source and
            the sink, NOT their IDs. If ``None``, the factors for all source-sink
            combinations are computed).
            
        """
        raise NotImplementedError()
    
    @inherit_doc(get_mean_factor)
    def get_mean_factor_autograd(self, parameters, parametersConsidered):
        """Same as :py:meth:`get_mean_factor`, but must use autograd's functions 
        instead of numpy. 
        
        This function is necessary to compute derivatives with automatic 
        differentiation.
        
        ~+~
        
        To write this function, copy the content of :py:meth:`get_mean_factor` 
        and exchange ``np.[...]`` with ``ag.[...]``
        
        .. note:: Autograd functions do not support in-place operations. 
            Therefore, an autograd-compatible implementation may be less efficient.
            If efficiency is not of greater concern, just use the autograd functions
            in `get_mean_factor` already and leave this method untouched.
            
        """
        return self.get_mean_factor(parameters, parametersConsidered)
    
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
    
    @inherit_doc(TransportNetwork)
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
        self.__erase_flow_model_fit()
    
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
        self.__erase_flow_model_fit()
            
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
        self.__erase_flow_model_fit()
        self.__erase_traffic_factor_model()
        if ("roadNetwork" in self.__dict__):
            self.originData = popData[self.roadNetwork.sourcesConsidered]
    
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
                
        
        destinationData = np.genfromtxt(fileNameDestinations, delimiter=",", skip_header = True, 
                                 dtype = dtype)
        destinationData = self.__trafficFactorModel_class.process_sink_covariates(destinationData)
        destinationData.sort(order="destinationID")

        
        self.rawDestinationData = destinationData
        self.__check_destination_road_match()
        self.__erase_flow_model_fit()
        self.__erase_traffic_factor_model()
        
        if ("roadNetwork" in self.__dict__ 
                and "sinksConsidered" in self.roadNetwork.__dict__):
            self.destinationData = self.rawDestinationData[self.roadNetwork.sinksConsidered]
    
    
    def set_infested(self, originID, infested=True):
        """Chenges the infestation state of an origin with the given ID.
        
        Parameters
        ----------
        originID : :py:data:`IDTYPE` 
            ID of the origin whose state shall be changed
        infested : bool
            Infestation state. ``True`` means infested.
            
        """
        inds = np.nonzero(self.originData["originID"]==originID)[0]
        if not inds.size:
            raise ValueError("A jursidiction with ID {} does not exist".format(originID))
        self.originData["infested"][inds] = infested
        
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
            destinationData = self.rawDestinationData
            if (not destinationData.size 
                    == self.roadNetwork.rawSinkIndexToVertexIndex.size):
                raise ValueError("The numbers of destinations in the destination data " 
                                 + str(destinationData["destinationID"].size) 
                                 + " and the road network " 
                                 + str(self.roadNetwork.sinkIndexToSinkID.size) 
                                 + " do not match. Maybe some destination data are"
                                 + " missing?")
            L = self.roadNetwork._DESTINATION_SINK_LABEL
            for ID1, ID2 in zip(destinationData["destinationID"], 
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
                              "Previous results are therefore inconsistent " +
                              "and will be removed.")
                self.__erase_processed_survey_data()
            else:
                reset = False
        else:
            reset = True
        
        if reset:
            if "rawOriginData" in self.__dict__:
                self.originData = self.rawOriginData[
                                                        sourcesConsidered]
            if "rawDestinationData" in self.__dict__:
                self.destinationData = self.rawDestinationData[sinksConsidered]
                
    
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
            self.__erase_survey_data()
    
    @inherit_doc(TransportNetwork.find_potential_routes)
    def find_potential_routes(self, 
            stretchConstant=1.5, 
            localOptimalityConstant=.2, 
            acceptionFactor=0.667,
            rejectionFactor=1.333):
        """# Find potential vector routes."""
        
        self.roadNetwork.find_potential_routes(stretchConstant, 
                                                     localOptimalityConstant, 
                                                     acceptionFactor,
                                                     rejectionFactor)
        
            
    def __create_travel_time_model(self, index=None, longDist=True, 
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
                trafficData = self.surveyData["longDistTimeData"]
            else:
                trafficData = self.surveyData["restTimeData"]
        
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
                        
    @inherit_doc(__create_travel_time_model)
    def create_travel_time_model(self, parameters=None, fileName=None):
        """Create and fit the travel time model.
        
        Parameters
        ----------
        fileName : str
            If given, a plot with the density function of the distribution will
            be saved under the given name as pdf and png. 
            Do not include the file name extension.
        
        """
        self.prst("Fitting the travel time model")
        self.travelTimeModel = travelTimeModel = self.__create_travel_time_model(parameters=parameters)
        
        self.__erase_processed_survey_data()
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
            If a survey shift has been conducted without any agent being 
            observed, include at least one observation with origin and 
            destination left blank and ``relevant`` set to ``False``.
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
        self.__erase_processed_survey_data()
        self.__erase_flow_model_fit()
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
                    #if not self.originData[fromIndex]["infested"]:
                    #if self.originData[fromIndex]["infested"]:
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
        
        self.surveyData = {}
        
        self.surveyData["pruneStartTime"] = pruneStartTime
        self.surveyData["pruneEndTime"] = pruneEndTime
        self.surveyData["longDistTimeData"] = dict(longDistTimeData)
        self.surveyData["restTimeData"] = dict(restTimeData)
        self.surveyData["shiftData"] = shiftData.get_array()
        self.surveyData["dayData"] = dayData.get_array()
        _properDataRate = acceptedCount/(rejectedCount+acceptedCount)
        self.prst("Fraction of complete data:", _properDataRate)
        if properDataRate is None:
            self.properDataRate = _properDataRate
        else:
            self.properDataRate = properDataRate
            self.prst("Set properDataRate to given value", properDataRate)
        
        dayCountData = self.surveyData["dayData"]["countData"]
        for i, d in enumerate(dayCountData):
            dayCountData[i] = dict(d)
        
        self.surveyData["pairCountData"] = pairCountData
        
        self.prst("Daily boater counts and time data determined.")
    
    
    def __erase_flow_model_fit(self):
        """Resets the gravity model to an unfitted state."""
        self.__dict__.pop("flowModelData", None)
    
    def __erase_traffic_factor_model(self):
        """Erases the gravity model."""
        self.__dict__.pop("trafficFactorModel", None)
        self.__erase_flow_model_fit()
    
    def __erase_travel_time_model(self):
        """Erases the travel time model."""
        self.__erase_route_choice_model()
        self.__dict__.pop("travelTimeModel", None)
    
    def __erase_route_choice_model(self):
        """Erases the route choice model."""
        self.__dict__.pop("routeChoiceModel", None)
        self.__erase_traffic_factor_model()
            
    def __erase_survey_data(self):    
        """Erases the survey data."""
        self.__dict__.pop("surveyData", None)
        self.__erase_processed_survey_data()
        self.__erase_travel_time_model()
    
    def __erase_processed_survey_data(self):    
        """Erases the observation data prepared for the model fit."""
        self.__dict__.pop("processedSurveyData", None)
        self.__erase_route_choice_model()
        
    
    def create_route_choice_model(self, redo=False):
        """Creates and fits the route choice model.
        
        Parameters
        ----------
        redo : bool
            Whether the route choice model shall be refitted if it has been 
            fitted already. If set to ``True``, the previous fit will be ignored.
        
        """
    
        self.prst("Creating route choice model")
        
        if not redo and "routeChoiceModel" in self.__dict__:
            self.prst("Route model has already been created.")
            return False
        if "inspectedPotentialRoutes" not in self.roadNetwork.__dict__:
            warnings.warn("Route candidates must be computed before a route "
                          + "model can be created. Nothing has been done. Call"
                          + "model.find_potential_routes(...)")
            return False
            
        self.increase_print_level()
        shiftData = self.surveyData["shiftData"]
        
        self.prst("Preparing road model")
        
        routeChoiceModel = RouteChoiceModel(parentPrinter=self)
        routeChoiceModel.set_fitting_data(self.surveyData["dayData"], shiftData, 
                                          self.roadNetwork.inspectedPotentialRoutes, 
                                          self.roadNetwork.lengthsOfPotentialRoutes, 
                                          self.travelTimeModel,
                                          self.complianceRate, self.properDataRate)
        self.routeChoiceModel = routeChoiceModel
            
        self.decrease_print_level()
        return True    
            
        
    def preprocess_survey_data(self, redo=False):
        """Takes the raw survey data and preprocesses them for the model fit.
        
        Parameters
        ----------
        redo : bool
            Whether the task shall be repeated if it had been done before.
            If set to ``True``, the earlier result be ignored.
        
        """
        
        self.prst("Extrapolating the boater count data")
        if not redo and "processedSurveyData" in self.__dict__:
            self.prst("Count data have already been prepared.")
            return False
        if "inspectedPotentialRoutes" not in self.roadNetwork.__dict__:
            warnings.warn("Route candidates must be computed before a route "
                          + "model can be created. Nothing has been done. Call"
                          + "model.find_potential_routes(...)")
            return False
        
        self.increase_print_level()
        self.prst("Extrapolating the count data")
        sourceIndexToSourceID = self.roadNetwork.sourceIndexToSourceID
        sinkIndexToSinkID = self.roadNetwork.sinkIndexToSinkID
        stationIndexToStationID = self.roadNetwork.stationIndexToStationID
        
        self.increase_print_level()
        
        self.prst("Extrapolating boater count data")
        
        countDType = {"names":["pairIndex", "p_shift", "count"], 
                      'formats':[int, 'double', int]}
        fullCountData = FlexibleArray(10000, dtype=countDType)
        
        shiftDType = {"names":["p_shift", "usedStationIndex", "shiftStart",
                               "shiftEnd"], 
                      'formats':['double', int, float, float]}
        shiftData = np.empty(self.surveyData["shiftData"].size, dtype=shiftDType)
        
        noiseDType = {"names":["pairIndex", "p_shift", "count"], 
                      'formats':[int, 'double', int]}
        observedNoiseData = FlexibleArray(10000, dtype=noiseDType)
        
        
        inspectedRoutes = self.roadNetwork.inspectedPotentialRoutes
        routeLengths = self.roadNetwork.lengthsOfPotentialRoutes
        countData = self.surveyData["shiftData"]
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
        counter = Counter(self.surveyData["shiftData"].size, 0.01)
        
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
        
        self.processedSurveyData = {
            "fullCountData": fullCountData.get_array(),
            "consideredPathLengths": consideredPathLengths,
            "observedNoiseData": observedNoiseData.get_array(),
            "shiftData": shiftData,
            "stationData": stationData
            }
        self.decrease_print_level()
        self.decrease_print_level()
        return True
    
    #@staticmethod
    #@inherit_doc(BaseTrafficFactorModel.get_mean_factor)
    @staticmethod_inherit_doc(BaseTrafficFactorModel.get_mean_factor)
    def _get_k_value_static(parameters, parametersConsidered, trafficFactorModel, pair=None):
        """Returns the ``k`` parameter of the negative binomial distribution.
        
        The value is computed accoring to the gravity model implemented in the
        ``trafficFactorModel``.
        
        Note that in contrast to the ``trafficFactorModel``,  Similarly, ``parametersConsidered`` must have entries
        for these parameters, which will be assumed to be ``True``.
        
        Parameters
        ----------
        trafficFactorModel : :py:class:`BaseTrafficFactorModel`
            Traffic factor model that is to be used to compute the ``k`` value.
        parameters #
            >!The first two entries must refer to the proportionality constant 
            and the parameter ``q``, which is `1-mean/variance`. The remaining 
            parameters are used to compute the traffic factor.
        parametersConsidered #
            >!The first two entries must refer to the proportionality constant 
            and the parameter ``q`` and will be assumed to be ``True``.
        
        """
        q = parameters[1]   
        c0 = parameters[0] * (1-q) / q  # reparameterization k->mu
        return trafficFactorModel.get_mean_factor(parameters[2:], parametersConsidered[2:], 
                                                  pair) * c0
    
    @inherit_doc(_get_k_value_static)
    def _get_k_value(self, parameters, parametersConsidered, pair=None):
        """#Returns the ``k`` parameter of the negative binomial distribution."""
        return HybridVectorModel._get_k_value_static(
            parameters, parametersConsidered, self.trafficFactorModel, pair)
    
    @staticmethod_inherit_doc(_get_k_value_static)
    def _get_k_value_autograd_static(parameters, parametersConsidered, trafficFactorModel):
        """Same as :py:meth:`_get_k_value_static`, but must use autograd's 
        functions instead of numpy. """
        
        q = parameters[1]   
        c0 = parameters[0] * (1-q) / q  # reparameterization k->mu
        return trafficFactorModel.get_mean_factor_autograd(parameters[2:], 
                                                           parametersConsidered[2:]) * c0
    
    @staticmethod_inherit_doc(BaseTrafficFactorModel.convert_parameters)
    def _convert_parameters_static(parameters, parametersConsidered, trafficFactorModel):
        """
        Parameters
        ----------
        parameters : float[] 
            Free parameters. The parameters that are not held constant.
        trafficFactorModel : :py:class:`BaseTrafficFactorModel`
            Traffic factor model that is to be used.
        
        """
        
        return ([convert_R_pos(parameters[0]), convert_R_0_1(parameters[1])] 
                 + trafficFactorModel.convert_parameters(parameters[2:], 
                                                         parametersConsidered[2:]))
    
    @inherit_doc(_convert_parameters_static)
    def _convert_parameters(self, parameters, parametersConsidered):
        return HybridVectorModel._convert_parameters_static(
            parameters, parametersConsidered, self.trafficFactorModel)
        
    #@staticmethod
    #@inherit_doc(_get_k_value_static)
    @staticmethod_inherit_doc(_get_k_value_static)
    def _negLogLikelihood(parameters, routeChoiceParameters, parametersConsidered, 
                          pairIndices, stationPairIndices, observedNoisePairs,
                          routeProbabilities, 
                          stationRouteProbabilities,
                          stationIndices,
                          p_shift, shiftDataP_shift, observedNoiseP_shift, 
                          p_shift_mean, stationKs, 
                          kSumNotObserved,
                          approximationNumber,
                          counts, observedNoiseCounts, trafficFactorModel): 
        """Returns the negative log-likelihood of the model.
        
        Parameters
        ----------
        routeChoiceParameters : float[]
            Route choice parameters. The first entry is the probability to 
            select an inadmissible path. The second entry is the exponent
            controlling the preference for shorter paths. The third entry is
            the probability that a given suvey location is on a randomly 
            selected inadmissible path.
        pairIndices : int[]
            For each set of agents who were (1) suveyed during the 
            same survey shift and (2) travelling between the smae 
            origin-destination pair the index of the origin-destination pair.
            The index for an origin-dedtination pair 
            `(fromIndex, toIndex)` is computed as 
            `fromIndex * #destinations + toIndex`.
        stationPairIndices : int[][]
            ``stationPairIndices[i]`` contains an int[] with the indices of pairs 
            that have an admissible path via survey station ``i``.
        observedNoisePairs : int[]
            For each agent that has been observed at a location that it is not
            on any admissible path between the agent's origin and destination,
            :py:obj:`observedNoisePairs` contains the repsective origin-destination
            pair index.
        routeProbabilities : float[]
            Contains for each agent set (see :py:obj:`pairIndices`)
            the probability that any given agent from this set 
            chooses a route via the location where they were observed. 
        stationRouteProbabilities : float[][]
            ``stationRouteProbabilities[i]`` contains a float[] with the 
            respective probabilities to drive via survey station i for all 
            origin-destination pairs in ``stationPairIndices``. That is,
            ``stationRouteProbabilities[i][j]`` is the probability that an agent
            will travel via location ``i`` on their trip between 
            origin-dsetination pair ``j``. 
        stationIndices : int[]
            Contains for each survey shift the index of the location where the
            survey was conducted.
        p_shift : float[]
            Contains for each agent set (see :py:obj:`pairIndices`)
            the probability that they timed their journey in a way that they 
            would pass the survey location while a survey was conducted there.
        shiftDataP_shift : float[]
            Contains for each survey shift the probability that an agent that is
            known to pass the respective survey location at some time does
            so while the survey was conducted.
        observedNoiseP_shift : float[]
            Contains for each set of agents that who was (1) surveyed in the
            same survey shift and (2) `not` observed on an
            admissible route between their origin and destination 
            the probability that they timed their journey in a way that they 
            would pass the survey location while a survey was conducted there.
        p_shift_mean : float
            Mean of :py:obj:`shiftDataP_shift`.
        stationKs : float[][]
            An empty array of dimension `(stationNumber, approximationNumber+1)`,
            whereby `stationNumber` is the number of used survey locations and
            approximationNumber the degree of the Tailor approximation that is
            to be used (see below).
        kSumNotObserved : float[]
            An empty array whose length coincides with the number of used
            survey locations.
        approximationNumber : int
            Degree of the Taylor approximation that is to be used. The higher 
            this number the more precise the likelihood will be but also the
            longer will the computation take. Must be `>= 1`.
        counts : int[]
            The size of each agent set described in the explanation for
            :py:obj:`pairIndices`.
        observedNoiseCounts :
            The size of each agent set described in the explanation for 
            :py:obj:`observedNoiseP_shift` .
        trafficFactorModel : :py:class:`BaseTrafficFactorModel`
            Traffic factor model used to determine the strengths of the agent
            flows between the individual origin-destination pairs.
        
        ~+~
        
        The rationale for passing empty arrays to :py:meth:`_negLogLikelihood`
        is to save the computation time for object creation. This may be an
        unnecessary optimization.
        
        """
        parameters = HybridVectorModel._convert_parameters_static(
                                    parameters, parametersConsidered, trafficFactorModel)
        
        c1 = parameters[1]
        c2, _, c4 = routeChoiceParameters
        
        kMatrix = HybridVectorModel._get_k_value_static(
                    parameters, parametersConsidered, trafficFactorModel)
        
        
        kMatrix = kMatrix.ravel() 
        k = kMatrix[pairIndices]
        
        
        aq = routeProbabilities * p_shift * c1
        qqm = (1-c1)/(1-c1+aq) 
        
        # Liekelihood associated with observations > 0 on expected routes
        likelihoodOnWays = np.sum(nbinom.logpmf(counts, k, qqm), 0)
        
        for i, stPairIndices, sRP in zip(itercount(),
                                         stationPairIndices, 
                                         stationRouteProbabilities):
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
        
        
        result = (- likelihoodOnWays - likelihoodRestWays 
                  - likelihoodFalseWays - likelihoodNotObservedOnWays)
        
        if np.any(np.isnan(result)): 
            return np.inf 
        
        return result
    
    @staticmethod_inherit_doc(_negLogLikelihood)
    def _negLogLikelihood_autograd(parameters, routeChoiceParameters,
                                   parametersConsidered, pairIndices, 
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
        """Returns the negative log-likelihood of the model, thereby using the
        functions provided by :py:mod:`autograd`.
        
        """
        
        log = ag.log
        
        if convertParameters:
            parameters = HybridVectorModel._convert_parameters_static(
                                parameters, parametersConsidered, trafficFactorModel)
        else:
            parameters = parameters
        
        c1 = parameters[1]
        c2, _, c4 = routeChoiceParameters
            
        kMatrix = HybridVectorModel._get_k_value_autograd_static(
                    parameters, parametersConsidered, trafficFactorModel)
        
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
        if isinstance(result, ag.float) and ag.isnan(result): 
            return ag.inf 
        
        return result

    @staticmethod_inherit_doc(_negLogLikelihood, set_compliance_rate)
    def _get_nLL_funs(processedSurveyData, lengthsOfPotentialRoutes, trafficFactorModel,
                      routeChoiceParameters, complianceRate, properDataRate,
                      parametersConsidered, approximationNumber=3):
        """Returns the model's negative log-likelihood function and its 
        derivatives.
        
        Parameters
        ----------
        processedSurveyData : dict
            Processed survey data, which are computed with :py:meth:`preprocess_survey_data`
        lengthsOfPotentialRoutes : :py:class:`csr_matrix_nd <vemomoto_core.npcollections.npext.csr_matrix_nd>`
            For each origin-destination pair the lengths of all potential
            (i.e. admissible) agent routes.
        properDataRate : float
            Fraction of agents providing inconsistent, incomplete, or wrong
            data.
        
        """
        
        if parametersConsidered is None:
            parametersConsidered = np.ones(20, dtype=bool)
        
        fullCountData = processedSurveyData["fullCountData"]
        consideredPathLengths = processedSurveyData["consideredPathLengths"] 
        observedNoiseData = processedSurveyData["observedNoiseData"] 
        shiftDataP_shift = processedSurveyData["shiftData"]["p_shift"] * complianceRate * properDataRate
        stationIndices = processedSurveyData["shiftData"]["usedStationIndex"]
        stationPathLengths = processedSurveyData["stationData"][
                                                        "consideredPathLengths"]
        stationPairIndices = processedSurveyData["stationData"]["pairIndices"]
        
        p_shift = fullCountData["p_shift"] * complianceRate * properDataRate
        pairIndices = fullCountData["pairIndex"]
        counts = fullCountData["count"]
        routeLengths = lengthsOfPotentialRoutes.data
        
        observedNoiseP_shift = observedNoiseData["p_shift"] * complianceRate * properDataRate
        observedNoiseCounts = observedNoiseData["count"]
        observedNoisePairs = observedNoiseData["pairIndex"]
        p_shift_mean = np.mean(shiftDataP_shift)
        stationNumber = stationPairIndices.size
        stationKs = np.zeros((stationNumber, approximationNumber+1))
        kSumNotObserved = np.zeros(stationNumber)
        
        _negLogLikelihood = HybridVectorModel._negLogLikelihood
        _negLogLikelihood_autograd = HybridVectorModel._negLogLikelihood_autograd
        
        
        c2, c3, c4 = routeChoiceParameters
        
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
        
        
        def negLogLikelihood(parameters): 
            return _negLogLikelihood(parameters, routeChoiceParameters, parametersConsidered, 
                   pairIndices, stationPairIndices, 
                   observedNoisePairs, routeProbabilities, 
                   stationRouteProbabilities, stationIndices, p_shift, 
                   shiftDataP_shift, observedNoiseP_shift, p_shift_mean, 
                   stationKs, kSumNotObserved, 
                   approximationNumber, counts, observedNoiseCounts, 
                   trafficFactorModel)
        
        def negLogLikelihood_autograd(parameters, convertParameters=True):
            return _negLogLikelihood_autograd(parameters, routeChoiceParameters,
                   parametersConsidered, pairIndices, stationPairIndices, 
                   observedNoisePairs, routeProbabilities, 
                   stationRouteProbabilities, stationIndices, p_shift, 
                   shiftDataP_shift, observedNoiseP_shift, p_shift_mean, 
                   stationNumber, approximationNumber, counts, 
                   observedNoiseCounts, trafficFactorModel,
                   convertParameters=convertParameters)
        
        jac = grad(negLogLikelihood_autograd)
        hess = hessian(negLogLikelihood_autograd)

        return negLogLikelihood, negLogLikelihood_autograd, jac, hess, None
    
    @staticmethod_inherit_doc(_get_nLL_funs)
    def maximize_log_likelihood_static(processedSurveyData, 
                                       lengthsOfPotentialRoutes,
                                       trafficFactorModel,
                                       routeChoiceParameters,
                                       complianceRate,
                                       properDataRate,
                                       parametersConsidered, 
                                       approximationNumber=3,
                                       flowParameters=None,
                                       x0=None):
        """Maximizes the likelihood of the hybrid model.
        
        Parameters
        ----------
        flowParameters : float[]
            If given, a model with these parameters will be used and assumed
            as being the best-fitting model.
        x0 : float[]
            Will be used as initial guess if given and if
            :py:obj:`flowParameters` is not given.
        
        """
        
        
        negLogLikelihood, negLogLikelihood_autograd, jac, hess, hessp = \
                HybridVectorModel._get_nLL_funs(processedSurveyData, 
                                                lengthsOfPotentialRoutes, 
                                                      trafficFactorModel, 
                                                      routeChoiceParameters,
                                                      complianceRate,
                                                      properDataRate,
                                                      parametersConsidered, 
                                                      approximationNumber)
        
        
        if flowParameters is None:
            
            bounds = [(-15, 15), (-10, 0.5)]
            
            parametersConsidered[:2] = True
            
            for bound in trafficFactorModel.BOUNDS[parametersConsidered[2:]]:
                bounds.append(tuple(bound))
                
            if x0 is None:
                np.random.seed()
                
                x0 = np.ones(np.sum(parametersConsidered))
                negLogLikelihood(x0)
                negLogLikelihood_autograd(x0)
                
                result = op.differential_evolution(negLogLikelihood, bounds, 
                                                   popsize=20, maxiter=20, #300, 
                                                   #popsize=20, maxiter=2, 
                                                   disp=True)
                print(parametersConsidered)          
                print("GA result", result)
                x0 = result.x.copy()
                result.xOriginal = HybridVectorModel._convert_parameters_static(
                    result.x, parametersConsidered, trafficFactorModel)
                result.jacOriginal = jac(result.xOriginal, False)
            else:
                result = op.OptimizeResult(x=x0, 
                                           success=True, status=0,
                                           fun=negLogLikelihood(x0), 
                                           nfev=1, njev=0,
                                           nhev=0, nit=0,
                                           message="parameters checked")
                result.xOriginal = HybridVectorModel._convert_parameters_static(
                    result.x, parametersConsidered, trafficFactorModel)
                result.jacOriginal = jac(result.xOriginal, False)
            
            
            
            print(negLogLikelihood(x0))
            print(negLogLikelihood_autograd(x0))
            
            result2 = op.minimize(negLogLikelihood_autograd, x0, method="L-BFGS-B",
                                  jac=jac, hess=hess,
                                  bounds=None, options={"maxiter":800,
                                                        "iprint":2})
            print(parametersConsidered)          
            result2.xOriginal = HybridVectorModel._convert_parameters_static(
                    result2.x, parametersConsidered, trafficFactorModel)
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
            print(parametersConsidered)          
            result2.xOriginal = HybridVectorModel._convert_parameters_static(
                    result2.x, parametersConsidered, trafficFactorModel)
            result2.jacOriginal = jac(result2.xOriginal, False)
            print("SLSQP", result2)         
            if result2.fun < result.fun:
                x0 = result2.x.copy()
                result = result2
            
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
                
            print(parametersConsidered)          
            result2.xOriginal = HybridVectorModel._convert_parameters_static(
                    result2.x, parametersConsidered, trafficFactorModel)
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
                flowParameters, parametersConsidered, trafficFactorModel)
            result.jacOriginal = jac(result.xOriginal, False)
            
            checkParametersO = HybridVectorModel._convert_parameters_static(
                flowParameters, parametersConsidered, trafficFactorModel)
        
        return result
    
    @inherit_doc(maximize_log_likelihood_static)
    def maximize_log_likelihood(self, parametersConsidered=None, approximationNumber=3,
                                flowParameters=None, x0=None):
        """# Maximizes the likelihood of the hybrid model"""
        routeChoiceParameters = self.routeChoiceModel.parameters
        
        return HybridVectorModel.maximize_log_likelihood_static(
                self.processedSurveyData, 
                self.roadNetwork.lengthsOfPotentialRoutes,
                self.trafficFactorModel, routeChoiceParameters,
                self.complianceRate, self.properDataRate, parametersConsidered, 
                approximationNumber, flowParameters, x0)
    
    
    @staticmethod_inherit_doc(_get_nLL_funs, find_profile_CI_bound)
    def _find_profile_CI_static(processedSurveyData, lengthsOfPotentialRoutes, 
                                trafficFactorModel, routeChoiceParameters,
                                complianceRate,
                                properDataRate,
                                parametersConsidered, 
                                index, x0, direction,
                                approximationNumber=3, 
                                profile_LL_args={}):
        """Searches the profile likelihood confidence interval for a given
        parameter.
        
        Parameters
        ----------
        profile_LL_args : dict
            Keyword arguments to be passed to :py:meth:`find_profile_CI_bound`.
        
        """
        
        negLogLikelihood, negLogLikelihood_autograd, jac, hess, hessp = \
                HybridVectorModel._get_nLL_funs(processedSurveyData, 
                                                lengthsOfPotentialRoutes, 
                                                      trafficFactorModel, 
                                                      routeChoiceParameters,
                                                      complianceRate,
                                                      properDataRate,
                                                      parametersConsidered, 
                                                      approximationNumber) 
        
        negLogLikelihood_autograd_ = lambda x: -negLogLikelihood_autograd(x)   
        jac_ = lambda x: -jac(x)   
        hess_ = lambda x: -hess(x)   
        
        return find_profile_CI_bound(index, direction, x0, negLogLikelihood_autograd_, jac_, hess_, 
                                     **profile_LL_args)
    
    
    @inherit_doc(_find_profile_CI_static)
    def investigate_profile_likelihood(self, x0, processedSurveyData, 
                                       lengthsOfPotentialRoutes, 
                                       trafficFactorModel, routeChoiceParameters,
                                       complianceRate,
                                       properDataRate,
                                       parametersConsidered, 
                                       approximationNumber=3,
                                       **profile_LL_args):
        """# Searches the profile likelihood confidence interval for a given
        parameter."""
        
        negLogLikelihood, negLogLikelihood_autograd, jac, hess, hessp = \
                HybridVectorModel._get_nLL_funs(processedSurveyData, 
                                                lengthsOfPotentialRoutes, 
                                                      trafficFactorModel, 
                                                      routeChoiceParameters,
                                                      complianceRate,
                                                      properDataRate,
                                                      parametersConsidered, 
                                                      approximationNumber) 
        
        self.prst("Investigating the profile likelihood")
        
        self.increase_print_level()
        
        if not "fun0" in profile_LL_args:
            self.prst("Determining logLikelihood")
            profile_LL_args["fun0"] = -negLogLikelihood_autograd(x0)
        if not "hess0" in profile_LL_args:
            self.prst("Determining Hessian of logLikelihood")
            profile_LL_args["hess0"] = -hess(x0)
        
        dim = len(x0)
        
        result = np.zeros((dim, 2))
        
        labels = ["c0", "q"] + list(self.trafficFactorModel.LABELS[parametersConsidered[2:]])
        
        indices, directions = zip(*iterproduct(range(dim), (-1, 1)))
        
        
        const_args = [processedSurveyData, lengthsOfPotentialRoutes, trafficFactorModel, 
                      routeChoiceParameters, self.complianceRate, self.properDataRate,
                      parametersConsidered]
        
        self.prst("Creating confidence intervals")
        #try: #, max_workers=13
        with ProcessPoolExecutor(const_args=const_args) as pool:
            mapObj = pool.map(HybridVectorModel._find_profile_CI_static, 
                              indices, repeat(x0), directions, 
                              repeat(approximationNumber), repeat(profile_LL_args))
            
            
            for index, direction, r in zip(indices, directions, mapObj):
                result[index][(0 if direction==-1 else 1)
                              ] = np.array(self._convert_parameters(r.x, 
                                        parametersConsidered))[parametersConsidered][index]
        
        self.prst("Printing confidence intervals and creating profile plots")
        self.increase_print_level()
        
        x0Orig = np.array(self._convert_parameters(x0, parametersConsidered))[parametersConsidered]
        
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
        self.decrease_print_level()
    
    @inherit_doc(RouteChoiceModel.fit)
    def fit_route_choice_model(self, refit=False, guess=None, 
                               improveGuess=False, disp=True, get_CI=True):
        """Fits the route choice model.
        
        Parameters
        ----------
        refit : bool
            Whether the model shall be refitted if it has already been fitted 
            earlier.
        get_CI : bool
            Whether confidence intervals for the parameters shall be computed
            after the model has been fitted.
        
        """
        
        self.increase_print_level()
        
        if "routeChoiceModel" not in self.__dict__:
            warnings.warn("A route choice model must be created before it can "
                          "be fitted. Call create_route_choice_model!")
            return False
        if not refit and self.routeChoiceModel.fitted:
            self.prst("A route choice model does already exist. I skip",
                      "this step. Enforce fitting with the argument",
                      "refit=True")
            return False
        if not self.routeChoiceModel.prepared:
            self.prst("The route choice model has not been prepared for model",
                      "fit. I ignore it.",
                      "Call create_route_choice_model if you want to",
                      "use the model.")
            return False
        
        self.prst("Fitting route choice model")
        
        self.increase_print_level()
        self.routeChoiceModel.fit(guess, improveGuess, disp)
        self.decrease_print_level()
        self.prst("Constructing confidence intervals for route",
                  "choice model")
        
        if not (guess is not None and improveGuess==False) and get_CI:
            self.increase_print_level()
            fileName = self.fileName
            if not os.access(fileName, os.F_OK): os.makedirs(fileName)
            fileName = os.path.join(fileName, fileName)
            self.routeChoiceModel.get_confidence_intervals(fileName)
            self.decrease_print_level()
        self.decrease_print_level()
        
        return True
    
    
    def set_traffic_factor_model_class(self, trafficFactorModel_class=None):
        """Sets the class representing the traffic factor (gravity) model.
        
        Parameters
        ----------
        trafficFactorModel_class : class
            Class of the traffic factor model. Must inherit from 
            :py:class:`BaseTrafficFactorModel`.
        
        """
        if trafficFactorModel_class is not None:
            if self.__dict__.pop("__trafficFactorModel_class", None) == trafficFactorModel_class:
                return
            trafficFactorModel_class._check_integrity()
            self.__trafficFactorModel_class = trafficFactorModel_class
        self.__dict__.pop("destinationData", None)
        self.__dict__.pop("rawDestinationData", None)
        self.__dict__.pop("originData", None)
        self.__dict__.pop("rawOriginData", None)
        self.__erase_traffic_factor_model()
    
    def prepare_traffic_factor_model(self):
        """Prepares the traffic factor model.
        
        This may be necessary if derived covariates shall be used and these
        derived covariates do not depend on paramters that shall be fitted.
        """
        if not self.__trafficFactorModel_class:
            raise ValueError("__trafficFactorModel_class is not specified. Call " 
                             + "`model.set_traffic_factor_model_class(...)`")
            
        self.trafficFactorModel = self.__trafficFactorModel_class(
            self.originData, self.destinationData, self.postalCodeAreaData, 
            self.roadNetwork.shortestDistances, 
            self.roadNetwork.postalCodeDistances)
        
    def fit_flow_model(self, permutations=None, refit=False, 
                       flowParameters=None, continueFlowOptimization=False, 
                       get_CI=True):
        """Fits the traffic flow (gravity) model.
        
        Fits one or multiple candidates for the traffic flow model and selects
        the model with minimal AIC value. 
        
        Parameters
        ----------
        permutations : bool[][]
            Each row corresponds to a parameter combination of a models that 
            is to be considered. For each parameter that could be potentially 
            included, the row must contain a boolean value. Do only include 
            parameters included in the traffic factor model. If ``None``,
            the :py:attr:`PERMUTATIONS <BaseTrafficModel.PERMUTATIONS>` given
            in the traffic factor model class will be considered. If this 
            attribute is not implemented, only the full model will be considered.
        refit : bool
            Whether to repeat the fitting procedure if the model has been 
            fitted earlier.
        flowParameters : dict
            Dictionary with the keys ``"parametersConsidered"`` and ``"parameters"`` 
            that provides an initial guess for the optimization or the 
            corresponding solution. ``"parametersConsidered"`` contains a `bool[]` with 
            the considered parameter combination (see :py:obj:`permutations`);
            ``"parameters"`` contains a `float[]` with the values for the 
            parameters where ``flowParameters["parametersConsidered"]`` is ``True``.
        continueFlowOptimization : bool
            If ``True``, the :py:obj:`flowParameters` will be used as initial 
            guess. Otherwise, they will be considered as the optimal 
            parameters.
        get_CI : bool
            Whether confidence intervals shall be computed after the model
            has been fitted. Note that no confidence intervals will be computed,
            if ``continueFlowOptimization is False``.
        
        """
        self.prst("Fitting flow models.")
        self.increase_print_level()
        
        fittedModel = False
        if not refit and "flowModelData" in self.__dict__:
            self.prst("A model does already exist. I skip",
                      "this step. Enforce fitting with the argument",
                      "refit=True")
            return False
        if "processedSurveyData" not in self.__dict__:
            self.prst("The model has no extrapolated boater data. I stop.",
                      "Call preprocess_survey_data if you want to",
                      "use the model.")
            return False
        if "trafficFactorModel" not in self.__dict__:
            self.prst("No traffic factor model has been specified. Call "
                      "model.set_traffic_factor_model(...)!")
            return False
        
        
        self.prst("Fitting the traffic factor model")
        self.increase_print_level()
        
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
        
        if flowParameters is None:
            
            if ("flowModelData" in self.__dict__  
                    and "parameters" in self.flowModelData 
                    and "covariates" in self.flowModelData
                    and "AIC" not in self.flowModelData):
                x0 = [self.flowModelData["parameters"]]
                permutations = np.array([self.flowModelData["parametersConsidered"]])
            else: 
                x0 = repeat(None)
            
            const_args = [self.processedSurveyData, 
                          self.roadNetwork.lengthsOfPotentialRoutes,
                          self.trafficFactorModel, 
                          self.routeChoiceModel.parameters, 
                          self.complianceRate, self.properDataRate]
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
            fittedModel = True
        else:
            if continueFlowOptimization:
                result = self.maximize_log_likelihood(
                                  flowParameters["parametersConsidered"], 
                                  x0=flowParameters["paramters"])
                parameters = [result.x]
                fittedModel = True
            else:
                result = self.maximize_log_likelihood(
                                  flowParameters["parametersConsidered"], 
                                  flowParameters=flowParameters["paramters"])
                parameters = [flowParameters["paramters"]]
            nLL = result.fun
            LLs.append(nLL)
            AICs = [2 * (np.sum(flowParameters["parametersConsidered"]) + nLL)]
            permutations = [flowParameters["parametersConsidered"]]
            results.append(result)
        
        self.decrease_print_level()
        
        bestIndex = np.argmin(AICs)
        AIC = AICs[bestIndex]
        LL = LLs[bestIndex]
        parameters = parameters[bestIndex]
        covariates = permutations[bestIndex]
        
        self.prst("Choose the following covariates:")
        self.prst(covariates)
        self.prst("Parameters (transformed):")
        self.prst(parameters) 
        self.prst("Parameters (original):")
        self.prst(results[bestIndex].xOriginal)
        self.prst("Negative log-likelihood:", LL, "AIC:", AIC)
        
        if results and fittedModel and get_CI: # or True:
            self.investigate_profile_likelihood(parameters,
                                            self.processedSurveyData, 
                                            self.roadNetwork.lengthsOfPotentialRoutes, 
                                            self.trafficFactorModel, 
                                            self.routeChoiceModel.parameters, 
                                            self.complianceRate,
                                            self.properDataRate,
                                            covariates, 
                                            disp=True, vm=False)
                 
            
        if ("flowModelData" not in self.__dict__ or
            "AIC" not in self.flowModelData or self.flowModelData["AIC"] >= AIC
            or (flowParameters is not None and not continueFlowOptimization)):
            self.flowModelData = {
                "AIC": AIC,
                "parameters": parameters,
                "parametersConsidered": covariates
                }
                    
        self.decrease_print_level()
        return fittedModel
        
    def get_station_mean_variance(self, stationIndices=None,
                                  shiftStart=0, shiftEnd=24,
                                  getStationResults=True,
                                  getPairResults=False,
                                  fullCompliance=False,
                                  correctData=False):
        """Returns the mean agent traffic that could be observed at survey 
        locations and the respective vairances. 
        
        The values are returned both for all agents and the agents
        coming from infested origins only, respectively. 
        
        The traffic can be returned either per survey location or per 
        origin-destination pair (assuming that surveys were conducted at the
        given locations and time intervals). 
        
        Parameters
        ----------
        stationIndices : int[]
            Indices of the locations for which the traffic estimate is desired.
            If ``None``, the traffic for all potential survey locations will
            be returned. The same location can be mentioned multiple times to 
            model multiple inspection shifts on different days.
        shiftStart : [0, 24)
            The start of the time interval for which the agent counts shall be
            estimated. Must be given in a 24h format. That is, 14.5 represents
            2:30PM. Can also be an array, which then must have the same length 
            as :py:obj:`stationIndices`.
        shiftStart : [0, 24)
            The end of the time interval for which the agent counts shall be
            estimated. Must be given in a 24h format. That is, 14.5 represents
            2:30PM. Can also be an array, which then must have the same length 
            as :py:obj:`stationIndices`.
        getStationResults : bool
            Whether the estimates shall be returned by survey location.
        getPairResults : bool
            Whether estimates shall be returned by origin-destination pair.
        
        ~+~
        
        .. todo:: This method can be made much more efficient, if the road 
            choice probabilities and the k values are computed once for each 
            origin-destination pair or inspection location only. That is,
            we would not need to reconsider teh same location multiple times.
            This would speed up this method by orders of magnitude.
        
        """
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
            
        c2, c3, c4 = self.routeChoiceModel.parameters
        
        infested = self.originData["infested"]
        
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
        
        routeLengthsPowers = copy(self.roadNetwork.lengthsOfPotentialRoutes)
        routeLengthsPowers.data = routeLengthsPowers.data.power(c3)
        routeLengthsNorm = routeLengthsPowers.data.sum(1).reshape(kMatrix.shape)
        inspectedRoutes = self.roadNetwork.inspectedPotentialRoutes
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
        """A subroutine of the algorithm for :py:meth:`get_station_mean_variance`
        for parallelization.
        
        .. todo:: Argument documentation
        
        """
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
        
    @inherit_doc(_get_k_value)
    def _get_k_q(self, pair=None, shiftStart=None, shiftEnd=None, 
                 stationIndex=None):
        """Computes the parameters `k` and `q` for the (negative binomial)
        traffic distribution between origin-destination pairs.
        
        The arguments can also be provided as arrays of matching dimensions.
        
        Parameters
        ----------
        shiftStart : [0, 24)
            The start of the time interval(s) for which the parameters shall be
            computed. Must be given in a 24h format. That is, 14.5 represents
            2:30PM. If not given, travel timing will be neglected and the 
            complete daily traffic flow will be considered.
        shiftStart : [0, 24)
            The end of the time interval(s) for which the parameters shall be
            computed. Must be given in a 24h format. That is, 14.5 represents
            2:30PM. If not given, travel timing will be neglected and the 
            complete daily traffic flow will be considered.
        stationIndex : int
            Index of the survey location to be considered. If not given,
            route choice will be neglected and the complete traffic flow will
            be computed.
        
        """
        covariates = self.flowModelData["parametersConsidered"]
        parameters = self._convert_parameters(self.flowModelData["parameters"], 
                                              covariates)
        q = parameters[1]
        k = self._get_k_value(parameters, covariates, pair)
        
        factor = 1
        if stationIndex is not None:
            c2, c3, c4 = self.routeChoiceModel.parameters
            try:
                routeLengths = self.roadNetwork.lengthsOfPotentialRoutes[pair].data
                stationPathLengths = [routeLengths[i] for i in 
                                      self.roadNetwork.inspectedPotentialRoutes[stationIndex][pair]]
                
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
        """Computes the optimal locations for agent inspections.
        
        Maximizes the number of agents who are inspected at least once given a 
        certain budget and other constraints for operation.
        
        The inspections are assumed to be conducted in shifts of given lengths.
        The best results will be obtained, if the number of possible shifts per
        day is an integer. However, other shift lengths are possible as well.
        
        The day will need to be discretized. The method will take efforts to 
        make the input match a discretization scheme so that not more time
        intervals than necessary need to be considered.   
        
        .. note:: This method assumes that `MOSEK <https://www.mosek.com/documentation/>`_
            is installed to solve linear programming problems. (See also the 
            `cvxpy documentation <https://www.cvxpy.org/install/>`_.)
            A different solver could be used as well, but this has to be changed
            in the source code.
        .. note:: By the time when this document was created, the MOSEK interface
            of cvxpy did not implement the option to pass an initial condition
            to the solver. If this feature shall be used (which is recommended),
            the cvxpy installation needs to be pached. Please copy the files
            in the subdirectory cvxpy_changes to the locations designated in 
            their headers and replace the original files.
        
        Parameters
        ----------
        costShift : float
            Costs per (daytime) inspection shift.
        costSite : float
            Costs per used inspection site.
        costBound : float
            Budget for the overall costs.
        shiftLength : int/float
            If given as `int`, length of an inspection shift measured in time steps;
            if given as `float`, approximate length of an inspection shift.
        nightPremium : (>=0, [0,24), [0,24))
            Describes additional costs for overnight inspections. Must be 
            given as a tuple ``(nightcost, start, end)``, whereby ``nightcost``
            is the cost for an overnight inspection shift, and ``start`` and 
            ``start`` and ``end`` denote the time interval in which the 
            additional costs are due (24h format). Note that ``nightcost`` 
            refers to a complete inspection shift conducted in the time interval 
            of additional costs. In practice, however, the costs will only 
            increased for the fraction of a shift that overlaps with the given 
            time interval of additional costs.
        allowedShifts : int[]/float[]
            List of permitted shift starting times measured in time units (if
            given as `int[]`) or as time points in the 24h format (if given as
            `float[]`).
        costRoundCoeff : float
            Specifies up to which fraction of the smallest cost the costs are 
            rounded. Rounding can increase the efficiency of the approach 
            significantly.
        baseTimeInv : int/float
            Number of time intervals per day (if given as `int`) or approximate
            length of one time interval in the 24h format (if given as `float`).
        ignoreRandomFlow : bool
            Indicates whether traffic via inadmissibe routes shall be ignored.
            This traffic noise adds uncertainty to the results but may lead to
            overall more precise estimates.
        integer : bool
            If ``True``, the solver applies an integer programming algorithm
            to solve the optimization problem. Otherwise a greedy rounding 
            scheme based on linear programming will be used (potentially faster 
            but with lower performance guarantee).
        timeout : float
            Timeout for internal optimization routines (in seconds).
        perturbation : float
            Perturbation added to make one inspection shift slightly more 
            effective. This is needed for tie breaking only.
        fancyRounding : bool
            If ``True`` a more sophisticated rounding scheme will be applied.
        full_result : bool
            If ``True``, the optimized variables will be returned in addition
            to the expected numer of inspected agents under the optimal 
            solution.
        extended_info : bool
            If ``True``, the covered fraction of the total agent flow and the
            used inspection locations (according to the optimal solution) will
            be returned in addition to other results.
        init_greedy : bool
            If ``True``, the greedy rounding algorithm will be used as initial 
            condition for the integer optimization algorithm. Use in conjunction
            with ``integer=True``.
        saveFile : bool
            Whether to save the optimization results to a file.
        loadFile : boold
            Whetehr the results may be loaded from a file if available.
        fileNameAddition : str
            Addition to the generated file name.
        
        """
        
        
        
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
        
        roadNetwork = self.roadNetwork
        
        stationCombinations = roadNetwork.stationCombinations
        stationIndexToRelevantStationIndex = {spot:i for i, spot in 
                        enumerate(roadNetwork.inspectedPotentialRoutes.keys())}
        relevantStationIndexToStationIndex = np.zeros(
                            len(stationIndexToRelevantStationIndex), dtype=int)
        
        for spot, i in stationIndexToRelevantStationIndex.items():
            relevantStationIndexToStationIndex[i] = spot
        
        originInfested = self.originData["infested"]
        covariates = self.flowModelData["parametersConsidered"] 
        parameters = self._convert_parameters(self.flowModelData["parameters"], covariates)
        routeLengths = roadNetwork.lengthsOfPotentialRoutes.data
        
        q = parameters[1]
        c2, c3, c4 = self.routeChoiceModel.parameters
        
        
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
        
        kArray = self._get_k_value(parameters, covariates)
        
        kArrayInfested = kArray[self.originData["infested"]]
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
                
                #print(operationResultReshaped[np.any(operationResultReshaped > 0, 1)])
                #print(np.array(spotusage.value)[np.any(operationResultReshaped > 0, 1)])
                
                if ((operationResult == 1) | (operationResult == 0)).all():
                    
                    # just because there could be multiple feasible solutions
                    spotusage.value = np.array(operations.value).reshape((spotNumber, shiftNumber)).max(1)
                    self.prst("Reached integer solution")
                    
                    break
                
                operations.value = (operationResult==1).astype(int)
                spotusageOriginal = np.array(spotusage.value).round(4)
                spotusage.value = np.array(operations.value).reshape((spotNumber, shiftNumber)).max(1)
                
                
                remainingBudgetValue = remainingBudget.value
                
                locationUsed |= np.array(spotusage.value).astype(bool).ravel()
                locationUsed_ = np.tile(locationUsed, (shiftNumber, 1)).T.ravel()
                affordable = (costShift__+(~locationUsed_)*costSite <= remainingBudgetValue)
                #covered = np.array(operatingTimeSums.value).astype(bool)
                covered = (matrixTimeConstraint.T@operationConstr + matrixTimeConstraint@operationConstr).astype(bool)
                mask = covered | (~affordable) | np.array(operations.value).astype(bool)
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
        
        if full_result:
            result = result, (np.array(flows.value), np.array(operations.value),
                              np.array(spotusage.value))
        
        if saveFile:
            saveobject.save_object(result, fileName)
        
        return result
    
    def create_caracteristic_plot(self, characteristic, values, 
                                  characteristicName=None, valueNames=None,
                                  **optim_kwargs):
        """Creates a plot of the characteristics of the optimal inspection
        policy.
        
        The probability that an agent chooses a time while the inspection 
        station is operated is plotted against the expected number of infested
        agents at the inspection stations for all used inspection stations.
        
        Parameters
        ----------
        characteristic : callable/str
            Property or argument whose impact on the results shall be studeied.
            If of type `callable`, for each entry ``val`` of :py:obj:`values`,
            ``callable(self, val)`` will be executed. If of type `str`, it will
            be interpreted as a keyword argument of 
            :py:obj:`optimize_inspection_station_operation`, which will be set
            to ``val``.
        values : arr
            Array of argument values.
        characteristicName : str
            Name of the property/argument that is studied. Used as axis label in
            the plot and to generate file names.
        valueNames : str
            Names of the values of the characteristic (used for the legend).
        **optim_kwargs : kwargs
            Keyword arguments passed to 
            :py:obj:`optimize_inspection_station_operation`.
        
        """
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
            
            self.prst("Covered flow", name, info["flowCover"])
            
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
        """Creates plots of the inspection success and price per inspected 
        agent dependent on the invested budget.
        
        Parameters
        ----------
        minBudget : float
            Minimal budget to be considered.
        maxBudget : float
            Maximal budget to be considered.
        nSteps : int
            Number of budget values to be considered.
        **optim_kwargs : kwargs
            Keyword arguments passed to 
            :py:obj:`optimize_inspection_station_operation`.
        
        """
        
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
        
    @inherit_doc(_get_k_q)
    def get_pair_distribution_property(self, dist_property=nbinom.mean, arg=None, 
                                      pair=None, shiftStart=None,
                                      shiftEnd=None):
        """Computes a property of the distribution of the agent flow 
        between origin-destination pairs. 
        
        Parameters
        ----------
        dist_property : callable
            The distribution property of interest. Must be properties 
            :py:obj:`scipy.stats.nbinom`. Can also be a list of 
            properties.
        arg : float
            Additional argument for :py:obj:`dist_property`. For example which
            quantile is desired, if `dist_property==nbinom.ppf`. Can also be
            of type `float[]`.
        pair : tuple
            The origin-destination pair(s) of interest as ``(fromIndex, toIndex)``
            respectively. If ``None``, the property will be computed for all
            origin-destination pairs.
        
        """
        
        k, q = self._get_k_q(pair, shiftStart, shiftEnd)
        qm = 1-q
        if arg is None:
            if hasattr(dist_property, "__iter__"):
                return [m(k, qm) for m in dist_property]
            else:
                return dist_property(k, qm)
        else:
            if hasattr(dist_property, "__iter__"):
                if hasattr(arg, "__iter__"):
                    return [m(a, k, qm) if a is not None else m(k, qm) 
                            for m, a in zip(dist_property, arg)]
                else:
                    return [m(arg, k, qm) for m in dist_property]
            else:
                return dist_property(arg, k, qm)
    
    def get_station_observation_prediction(self, predictions=None):
        """Returns observed and predicted agent counts for the inspection
        locations for which data are available.
        
        Parameters
        ----------
        predictions : struct[]
            If the predictions have been computed earlier, they can be provided
            as this argument. Otherwise the predictions will be computed anew.
            Must be the results of :py:meth:`get_station_mean_variance`.
        
        """
        
        countData = self.surveyData["shiftData"]
            
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
        """Returns the mean observed and predicted agent counts at survey 
        locations, thereby scaling the values so that they should come from
        a normal distribution.
        
        Only data obtained between a specific daytime interval will be 
        considered to ensure individual observations are identically distributed
        and will yield a normal distribution when added together.
        
        
        minSampleSize : int
            The minimal number of survey shifts that must be available before
            a survey location can be considered. If this value is too low, the
            resulting values will not follow an approximate normal distribution.
            If the value is too large, no data will be available to compute the
            results.
            
        """
        countData = self.surveyData["shiftData"]
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
                stationIndices, self.surveyData["pruneStartTime"], 
                self.surveyData["pruneEndTime"])
        
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
    
    @inherit_doc(get_station_observation_prediction)
    def get_pair_observation_prediction(self, predictions=None):
        """Returnes predicted and observed agent counts by origin-destination 
        pair. 
        
        """
        if predictions is None:
            countData = self.surveyData["shiftData"]
            predictions = self.get_station_mean_variance(
                    countData["stationIndex"], countData["shiftStart"], 
                    countData["shiftEnd"], getStationResults=False,
                    getPairResults=True)
            
        dtype = {"names":["count", "mean", "variance"], 
                 'formats':[int, 'double', 'double']}
        result = np.zeros(self.surveyData["pairCountData"].shape, dtype=dtype)
        result["count"] = self.surveyData["pairCountData"]
        result["mean"] = predictions["mean"]
        result["variance"] = predictions["variance"]
        return result
    
    
    def check_count_distributions_NB(self, minDataSetSize=20, fileName=None):
        """Checks whether the observed data may be normally distributed.
        
        Computes p-values for the test with null hypothesis 
        `H0: Data are negative binomially distributed`. If the p-values are
        high, we cannot reject the hypothesis and may conclude that the negative
        binomial distribution is modelling the data appropriately.
        
        Parameters
        ----------
        minDataSetSize : int
            It is necessary that parts of the considered data are identically
            distributed and that the samples are large enough. :py:obj:`minDataSetSize` 
            sets the minimal size of such a data set.
        fileName : str
            Names of the files to which plots of the resulting p-values will be 
            saved.
        
        """
        self.prst("Checking whether the count data follow a negative "
                  + "binomial distribution")
        
        self.increase_print_level()
        
        allObservations = []
        allParams = []
        allCountData = self.surveyData["shiftData"][self.surveyData["shiftData"]["prunedCount"] >= 0]
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
            parameters = FlexibleArrayDict(
                (3*max(len(countData[0]), 100), 2)
                )
            
            for i, obsdict in enumerate(countData):
                for pair, count in obsdict.items():
                    if pair not in parameters.indexDict:
                        k, q = self._get_k_q(pair, 
                                 self.surveyData["pruneStartTime"], self.surveyData["pruneEndTime"], 
                                 stationIndex)
                        parameters.add(pair, [k, 1-q])
                    observations.get(pair)[i] = count
            
            allObservations.append(observations.get_array())
            allParams.append(parameters.get_array())
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
        
    
    def compare_travel_time_distributions(self, saveFileName=None):
        """Compares agents' travel time distributions at different survey
        locations.
        
        Conducts likelihood ratio tests evaluating whether multiple 
        distributions are equal and plots the respective best-fitting time
        distributions at different locations.
        
        Compares not only travel time distributions at different locations but 
        also travel time distributions of local and long-distance travellers.
        
        Parameters
        ----------
        saveFileName : str
            File name for plots.
        
        """
        
        
        self.prst("Creating traffic distribution comparison plots")
        self.increase_print_level()
        
        # create comparison plots
        times = np.linspace(0, 24, 5000)
        yLabel = "Traffic Density"
        xLabel = "Time"
        
        # compare long distance versus short distance
        plt.figure()
        longDistDistribution = self.__create_travel_time_model(None, True)
        restDistribution = self.__create_travel_time_model(None, False)
        H0Distribution = self.__create_travel_time_model(None, None,
                            {key:val for key, val 
                             in enumerate(iterchain((
                                 self.surveyData["longDistTimeData"].values()), 
                                 self.surveyData["restTimeData"].values()))}) #dict must be merged without overwriting keys!
        
        
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
                            self.surveyData["longDistTimeData"].items() if len(val)>=50}
        restTimeData = {key:val for key, val in 
                        self.surveyData["restTimeData"].items() if len(val)>=50}
        
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
                    dist = self.__create_travel_time_model(label, long)
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
                        dist = self.__create_travel_time_model([label, 
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
            self.prst(np.round(LR,3))
            self.prst(plh, "p-Value:")
            self.prst(np.round(p,3))
            
            someInfNan = not np.isfinite(nLL).all()
            if someInfNan:
                warnings.warn("Some likelihood values are NaNs or Infs. "+
                              "The following reuslt may be biased.")
                
            H0Distribution = self.__create_travel_time_model(None, None, 
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
    
    
    
    def get_PMF_observation_prediction(self, stationID, fromID, toID, xMax=None, 
                                       getBestPMF=True, 
                                       getPureObservations=False):
        """Determines the observed and predicted probability mass function for 
        agent counts between specific origin-destination pairs.
        
        Parameters
        ----------
        stationID : :py:data:`IDTYPE`
            ID of the survey location where the considered data have been 
            collected. Can be an array.
        fromID : :py:data:`IDTYPE`
            ID of the origin of the considered agents. Can be an array.
        toID : :py:data:`IDTYPE`
            ID of the destination of the considered agents. Can be an array.
        xMax : int
            Count value up to which the probablity mass function is plotted
            at least. If not given, the probability mass function will be 
            computed up to the maximal observed count value.
        getBestPMF : bool
            If ``True``, a negative binomial distribution will be fitted 
            directly to the observed data. This can be helpful if it is of
            interest whether the observations come from a negative binomial 
            distribution.
        getPureObservations : bool
            Whether the pure observed count data shall be returned in addition
            to the other results.
        
        """
        
        stationIndex = self.roadNetwork.stationIDToStationIndex[stationID]
        fromIndex = self.roadNetwork.sourceIDToSourceIndex[fromID]
        toIndex = self.roadNetwork.sinkIDToSinkIndex[toID]
        
        countData = self.surveyData["shiftData"]
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
                             self.surveyData["pruneStartTime"], self.surveyData["pruneEndTime"], 
                             stationIndex)
        
        if hasattr(k, "__iter__"): k = k[0]
        
        predictedPMF = nbinom.pmf(X, k, 1-q)
        
        result = (X, observedPMF, predictedPMF)
        
        if getBestPMF:
            def negLogLikelihood(parameters):
                kk, qq = parameters
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
    
    
    @inherit_doc(get_PMF_observation_prediction)
    def compare_distributions(self, stationID, fromID, toID, xMax=None, 
                              saveFileName=None):
        """Compares distributions of agent obervations via Anderson-Darling
        tests and comparative plots of the observed and
        predicted cumulitive mass functions.
        
        Parameters
        ----------
        saveFileName : str
            File name for plots. If ``None`` no plots will be saved.
        
        """
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
    
    @inherit_doc(get_normalized_observation_prediction)
    def test_1_1_regression(self, minSampleSize=20, saveFileName=None,
                            comparisonFileName=None):
        """Tests whether the model results are biased.
        
        Compares predicted and observed values and tests the null hypothesis
        that the model yields unbiased estimates. If we obtain a high p-value
        and are unable to reject this hypothesis, we may assume that the model
        dies a good job.
        
        Parameters
        ----------
        saveFileName : str
            File name for a plot depicting the test and to save the results.
        comparisonFileName : str
            File name to load results from a different model or different 
            data set for comparison.
        
        """
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
                      saveFileName=_non_join(saveFileName, "_scaled"),
                      comparisonFileName=_non_join(comparisonFileName, "_scaled")
                      )
        
        self.decrease_print_level()
    
    @inherit_doc(create_observed_predicted_mean_error_plot)
    def create_quality_plots(self, worstLabelNo=5,
                             saveFileName=None,
                             comparisonFileName=None):
        """Creates predicted vs. observed plots.
        
        Parameters
        ----------
        worstLabelNo : int
            Number of data points that shall be labelled with their respective 
            IDs. The data points will be labelled in order of the largest
            deviance between predictions and observations.
        
        ~+~
        
        .. todo:: Compute mean at stations only, incorporate timing later.
            This could speed up the procedure significatnly.
        
        """
        
        #!!!!!!!!!!!!!!! porting an old version of the program.
        if not hasattr(self, "originData"):
            self.originData = self.lakeData
            del self.lakeData 
            self.destinationData = self.jurisdictionData
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
        
        countData = self.surveyData["shiftData"]
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
            _non_join(saveFileName, "Stations"),
            )
        create_observed_predicted_mean_error_plot(
            stationData["mean"].ravel(),
            stationData["count"].ravel(),
            saveFileName=_non_join(saveFileName, "Stations_raw"),
            comparisonFileName=_non_join(comparisonFileName, "Stations_raw")
            )
        create_observed_predicted_mean_error_plot(
            stationData["mean"].ravel()/station_std,
            stationData["count"].ravel()/station_std,
            saveFileName=_non_join(saveFileName, "Stations_scaled"),
            comparisonFileName=_non_join(comparisonFileName, "Stations_scaled")
            )
        
        self.prst("Creating plot of the quality by pair.")
        pairData = self.get_pair_observation_prediction(rawPairData)
        pair_std = np.sqrt(pairData["variance"].ravel())
        create_observed_predicted_mean_error_plot(
            pairData["mean"].ravel(),
            pairData["count"].ravel(),
            pair_std,
            title="Predicted and observed boater flows by source-sink pair",
            saveFileName=_non_join(saveFileName, "Pairs")
            )
        create_observed_predicted_mean_error_plot(
            pairData["mean"].ravel(),
            pairData["count"].ravel(),
            saveFileName=_non_join(saveFileName, "Pairs_raw"),
            comparisonFileName=_non_join(comparisonFileName, "Pairs_raw")
            )
        create_observed_predicted_mean_error_plot(
            pairData["mean"].ravel()/pair_std,
            pairData["count"].ravel()/pair_std,
            saveFileName=_non_join(saveFileName, "Pairs_scaled"),
            comparisonFileName=_non_join(comparisonFileName, "Pairs_scaled")
            )
        
        self.prst("Creating plot of the quality by origin.")
        mean = np.sum(pairData["mean"], 0).ravel()
        count = np.sum(pairData["count"], 0).ravel()
        if worstLabelNo >= self.destinationData.size:
            labels = self.destinationData["destinationID"]
        else:
            diff = np.abs(mean-count)
            max10DiffInd = np.argpartition(diff, -worstLabelNo)[-worstLabelNo:]
            labels = np.empty_like(mean, dtype=object)
            labels[max10DiffInd] = self.destinationData["destinationID"][max10DiffInd]
        destination_std = np.sqrt(np.sum(pairData["variance"], 0)).ravel()
        create_observed_predicted_mean_error_plot(
            mean,
            count,
            destination_std,
            title="Predicted and observed boater flows by destination",
            labels=labels,
            saveFileName=_non_join(saveFileName, "Destinations")
            )
        create_observed_predicted_mean_error_plot(
            mean, count,
            saveFileName=_non_join(saveFileName, "Destinations_raw"),
            comparisonFileName=_non_join(comparisonFileName, "Destinations_raw")
            )
        create_observed_predicted_mean_error_plot(
            mean/destination_std, count/destination_std,
            saveFileName=_non_join(saveFileName, "Destinations_scaled"),
            comparisonFileName=_non_join(comparisonFileName, "Destinations_scaled")
            )
        
        self.prst("Creating plot of the quality by origin.")
        mean = np.sum(pairData["mean"], 1).ravel()
        count = np.sum(pairData["count"], 1).ravel()
        if worstLabelNo >= self.originData.size:
            labels = self.originData["originID"]
        else:
            diff = np.abs(mean-count)
            max10DiffInd = np.argpartition(diff, -worstLabelNo)[-worstLabelNo:]
            labels = np.empty_like(mean, dtype=object)
            labels[max10DiffInd] = self.originData["originID"][
                                                                max10DiffInd]
        jur_std = np.sqrt(np.sum(pairData["variance"], 1)).ravel()
        create_observed_predicted_mean_error_plot(
            mean,
            count,
            jur_std,
            title="Predicted and observed boater flows by origin",
            labels=labels,
            saveFileName=_non_join(saveFileName, "Origins")
            )
        create_observed_predicted_mean_error_plot(
            mean, count,
            saveFileName=_non_join(saveFileName, "Origins_raw"),
            comparisonFileName=_non_join(comparisonFileName, "Origins_raw")
            )
        create_observed_predicted_mean_error_plot(
            mean/jur_std, count/jur_std,
            saveFileName=_non_join(saveFileName, "Origins_scaled"),
            comparisonFileName=_non_join(comparisonFileName, "Origins_scaled")
            )
        
        """
        self.prst("Creating plot of the quality by pair (log scale).")
        create_observed_predicted_mean_error_plot(
            pairData["mean"].ravel(),
            pairData["count"].ravel(),
            np.sqrt(pairData["variance"].ravel()),
            title="Predicted and observed boater flows by source-sink pair",
            saveFileName=_non_join(saveFileName, "Pairs_logScale"),
            logScale=True
            )
        create_observed_predicted_mean_error_plot(
            pairData["mean"].ravel(), pairData["count"].ravel(),
            saveFileName=_non_join(saveFileName, "Pairs_logScale_raw"),
            logScale=True,
            comparisonFileName=_non_join(comparisonFileName, "Pairs_logScale_raw")
            )
        """
        
        plt.show()
    
    def save_model_predictions(self, fileName=None):
        """Computes and saves model predictions by origin, destination, 
        origin-destination pair, and inspection location.
        
        Saves the estimated mean traffic and 
        
        Parameters
        ----------
        fileName : str
            Base of the file name to which the predictions shall be saved.
        
        """
        
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
        pairData = self.get_pair_distribution_property(measures, args)
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
        destinationResult = np.zeros(len(toID), 
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
        
        assert (originResult["fromID"]==self.originData["originID"]).all()
        
        destinationResult["toID"] = result["toID"][0]
        destinationResult["mean"] = result["mean"].sum(0)        
        destinationResult["variance"] = result["variance"].sum(0)        
        destinationResult["meanInfested"] = result["mean"][self.originData["infested"]].sum(0)        
        destinationResult["varianceInfested"] = result["variance"][self.originData["infested"]].sum(0)        
        destinationResult["95_percentile"] = get_ppf(destinationResult["mean"], 
                                              destinationResult["variance"])
        destinationResult["95_percentileInfested"] = get_ppf(
            destinationResult["meanInfested"], destinationResult["varianceInfested"])
        
        
        df = pd.DataFrame(originResult)
        df.to_csv(fileName + "Origin.csv", index=False)
        df = pd.DataFrame(destinationResult)
        df.to_csv(fileName + "Destination.csv", index=False)
        
    
    @inherit_doc(_convert_parameters_static)
    def simulate_count_data(self, stationTimes, day, parameters, parametersConsidered,
                            limitToOneObservation=False):
        """Simulate observation data that would be obtained one one day if the 
        model were True.
        
        For each boater, start, end, and path are returned.
        
        Parameters
        ----------
        stationTimes : dict
            Keys: indices of the survey locations; values: 2-tuples for start 
            and end time of the survey shift at that location (in 24h format).
        day : int
            ID for that day.
        parameters : float[]
            Parameters for the traffic flow (gravity) model.
        limitToOneObservation : bool
            Whether an agent travelling on an inadmissible route can be observed 
            at one location only (as assumed when we fit the route choice model) 
            or at multiple locations.
        
        """
        routeLengths = self.roadNetwork.lengthsOfPotentialRoutes.data
        
        parameters = self._convert_parameters(parameters, parametersConsidered)
        
        pRandom, routeExp, pObserve = self.routeChoiceModel.parameters
        kMatrix = self._get_k_value(parameters, parametersConsidered) 
        
        q = parameters[1]
        
        
        # number of people going for each pair
        n1 = np.random.negative_binomial(kMatrix, 1-q)
        
        routePowers = sparsepower(routeLengths, routeExp)
        normConstants = sparsesum(routePowers)
        
        pairToPairIndex = self.roadNetwork._pair_to_pair_index
        
        multObservations = defaultdict(lambda: 0)
        
        stations = list(stationTimes.keys())
        times = list(stationTimes.values())
        shiftNo = len(stations)
        inspectedRoutes = self.roadNetwork.inspectedPotentialRoutes
        observationsDType = [
            ("stationID", IDTYPE),
            ("day", int),
            ("shiftStart", "double"),
            ("shiftEnd", "double"),
            ("time", "double"),
            ("sourceID", IDTYPE),
            ("sinkID", IDTYPE),
            ("relevant", bool),
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
                            if np.random.rand() < self.properDataRate:
                                sinkID = sinkIndexToSinkID[sink]
                            else:
                                sinkID = b''
                            tuples.append((
                                stationIndexToStationID[stations[locInd]],
                                day,
                                start,
                                end,
                                time,
                                sourceIndexToSourceID[source],
                                sinkID,
                                True
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
                b'',
                False
                ))
        
        observations = observations.get_array()
        observations.sort(order="stationID")
        
        return observations, multObservations
    
    @inherit_doc(simulate_count_data)
    def save_simulated_observations(self, parameters=None, parametersConsidered=None,
                                    shiftNumber=None,
                                    dayNumber=None,
                                    stationSets=None,
                                    fileName=None):
        """Simulate observation data that would be obtained if the model were 
        True.
        
        Parameters
        ----------
        shiftNumber : int
            Number of observation shifts to be considered.
        dayNumber : int
            Number of days on which the shifts were conducted.
        stationSets : int[][]
            Sets/lists of survey location IDs at which inspections could be
            conducted simultaneously.
        fileName : str
            Name of the file to which the generated observations shall be saved.
        
        """
        if fileName is None: 
            fileName = self.fileName
            
        if parameters is None:
            parameters = self.flowModelData["parameters"]
            parametersConsidered = self.flowModelData["parametersConsidered"]
        
        self.prst("Simulating observations for static parameters", 
                  parameters, "with considered parameters", parametersConsidered)    
        
        #self.prst(self.travelTimeModel.location, self.travelTimeModel.kappa)
        
        if not shiftNumber:
            shiftData = self.surveyData["shiftData"].copy()
        else:
            shiftData = np.zeros(shiftNumber, dtype=self.surveyData["shiftData"].dtype)
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
                                                    self.surveyData["shiftData"]["stationIndex"]), 
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
                parametersConsidered)
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
    
    @staticmethod_inherit_doc(read_origin_data, read_destination_data, read_survey_data,
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
        preprocessSurveyData: bool
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
            model.save()
        
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
            if ((not "surveyData" in attrDict 
                 or restartArgs["readSurveyData"])
                    and "roadNetwork" in attrDict
                    and fileNameObservations is not None):
                properDataRate = restartArgs.get("properDataRate", None)
                model.read_survey_data(fileNameObservations, 
                                            properDataRate=properDataRate)
                #model.save()
                
            if ((not "travelTimeModel" in attrDict 
                 or restartArgs["fitTravelTimeModel"])
                    and "surveyData" in attrDict):
                travelTimeParameters = restartArgs.get("travelTimeParameters", None)
                model.create_travel_time_model(travelTimeParameters, model.fileName)
                #model.save()
        else:
            if restart: model.save()
        
        if ("roadNetwork" in attrDict and routeParameters is not None):
            
            if restartArgs["preprocessSurveyData"]:
                model.__erase_processed_survey_data()
            
            if (model.roadNetwork.__dict__.get("inspectedPotentialRoutes", None) is None or 
                    restartArgs["findPotentialRoutes"]):
                model.find_potential_routes(*routeParameters)
                model.save()
            
            if destinationToDestination:
                raise NotImplementedError("A model with destination to destination traffic"
                                          + " has not yet been"
                                          + " implemented completely.")
            
            if ("travelTimeModel" in attrDict):
                save = model.create_route_choice_model(restartArgs["createRouteChoiceModel"])
                save = model.preprocess_survey_data() or save
                if save:
                    #model.save()
                    pass
            
                
            if ("travelTimeModel" in attrDict):
                
                save = False
                routeChoiceParameters = restartArgs.get("routeChoiceParameters", None)
                continueRouteChoiceOptimization = restartArgs[
                                            "continueRouteChoiceOptimization"]
                
                
                refit = restartArgs["fitRouteChoiceModel"]
                save = model.fit_route_choice_model(refit, routeChoiceParameters,
                                                    continueRouteChoiceOptimization)
                
                if save:
                    #model.save()
                    pass
                    
                flowParameters = restartArgs.get("flowParameters", None)
                continueTrafficFactorOptimization = restartArgs["continueTrafficFactorOptimization"]
                
                #if model.fit_flow_model(parameters, refit=refit, flowParameters=flowParameters):
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
            #model.create_quality_plots(parameters, 
            #                               model.fileName + str(parameters))
        return model    
            
