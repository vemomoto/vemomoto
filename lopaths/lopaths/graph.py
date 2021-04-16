'''
Module implementing objects representing graphs and different algorithms to
find shortest and potential alternative paths.

'''
from functools import partial
from itertools import product as iterproduct, repeat, starmap, count as itercount
from itertools import chain as iterchain
import warnings
from collections import deque, defaultdict
import os
import copy as cp
from copy import deepcopy
from multiprocessing import Pool, Queue, Manager, Array
import ctypes
from queue import Queue as ThreadQueue
import threading
from multiprocessing import SimpleQueue as MultiprocessingQueue
import timeit
import datetime
import time
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import numpy.lib.recfunctions as rfn
import sharedmem

from vemomoto_core.npcollections.FixedOrderedIntDict import FixedOrderedIntDict
from vemomoto_core.npcollections.intquickheapdict import intquickheapdict
from vemomoto_core.npcollections.npextc import FlexibleArray, FlexibleArrayDict, \
    unique_tol, find_next_nonzero2d
from vemomoto_core.npcollections.npext import fields_view, add_alias, add_names, \
    FlexibleArrayDict as FlexibleArrayDictO, csr_matrix_nd, merge_arrays

from vemomoto_core.tools.simprofile import profile
from vemomoto_core.tools.hrprint import HierarchichalPrinter
from vemomoto_core.tools.iterext import Repeater, DictIterator
from vemomoto_core.tools.tee import Tee
from vemomoto_core.tools.saveobject import SeparatelySaveable
from vemomoto_core.tools.doc_utils import DocMetaSuperclass
from vemomoto_core.concurrent.nicepar import getCounter, Counter, Lockable, \
    Locked, ParallelCounter, CircularParallelCounter
from vemomoto_core.concurrent.concurrent_futures_ext import \
    ProcessPoolExecutor as ProcessPoolExecutor_ext 

try:
    from .sig_fig_rounding import RoundToSigFigs_fp as round_rel
except ImportError:
    from sig_fig_rounding import RoundToSigFigs_fp as round_rel
try:
    from .graph_utils import find_shortest_distance, in_sets
except ImportError:
    from graph_utils import find_shortest_distance, in_sets

#profiling
try:
    import line_profiler
except ImportError:
    pass

if len(sys.argv) > 1:
    teeObject = Tee(sys.argv[1])



CPU_COUNT = os.cpu_count() 
"Number of CPU cores to be used at most. Defaults to the number of installed cores."


class FlexibleGraph(HierarchichalPrinter):
    '''
    Graph with a flexible structure that supports efficient addition and removal
    of vertices and edges.
    '''

    def __init__(self, edges, edgeData, vertices, vertexData, 
                 replacementMode="overwrite", lengthLabel=None,
                 significanceLabel=None,
                 defaultVertexData=None, defaultEdgeData=None, **printerArgs):
        """
        Construtor
        """
        
        super().__init__(**printerArgs)
        
        for obj, prop in ((edges, edgeData), 
                          (vertices, vertexData)):
            if not obj.shape[0] == prop.shape[0]:
                raise ValueError("Properties must have the same length " 
                                 + "as the list of object that they discribe.")
        
        if (significanceLabel and 
                significanceLabel not in edges.array.dtype.names):
            raise ValueError("There is no field " + str(lengthLabel) 
                             + " in the edge data array.")
        self.significanceLabel = significanceLabel
        
        overrideExisting = replacementMode == "override"
        ignoreNew = replacementMode == "ignore"
        useShortest = replacementMode == "shortest"
        useLongest = replacementMode == "longest"
        if not (overrideExisting or ignoreNew or useShortest or useLongest):
            raise ValueError("replacementMode not understood. It must be one "
                             + "of 'override', 'ignore', 'shortest', or "
                             + "'longest'.")
            
        
        vertexIDs = add_names(vertices, "ID")
        
        vertexData = rfn.merge_arrays((vertexIDs, vertexData), 
                                      flatten=True)
        vertexData.sort(order="ID")
        includedVertices = np.zeros(len(vertices), dtype=bool)
        
        vertexData = np.rec.array(vertexData, copy=False)
        
        edgeCount = 0
        
        if useShortest or useLongest:
            if not lengthLabel in edgeData.dtype.names:
                raise ValueError("lengthLabel must match one of the field "
                                 + "names of edgeData.")
            lengthArr = edgeData[lengthLabel]
        
        graph = self.graph = {}
        overwroteSome = False
        loopsOccurred = False
        
        for row, pair in enumerate(edges):
            if pair[0] == pair[1]:
                loopsOccurred = True
                continue
            for i, direction in enumerate((1, -1)):
                thisVertex, otherVertex = pair[::direction]
                try:
                    vertexDataList = graph[thisVertex]
                except KeyError:
                    vertexIndex = np.searchsorted(vertexData.ID, 
                                                  thisVertex)
                    try:
                        if not vertexData.ID[vertexIndex] == thisVertex:
                            vertexIndex = None
                    except IndexError:
                        vertexIndex = None
                        
                    graph[thisVertex] = vertexDataList = [vertexIndex, {}, {}]
                    if not vertexIndex is None:
                        includedVertices[vertexIndex] = True
                
                if otherVertex in vertexDataList[i+1]:
                    overwroteSome = True
                    if ((useShortest 
                         and lengthArr[vertexDataList[i+1][otherVertex]] 
                                > lengthArr[row])
                        or (useLongest
                            and lengthArr[vertexDataList[i+1][otherVertex]] 
                                < lengthArr[row]) 
                        or overrideExisting):
                        
                        vertexDataList[i+1][otherVertex] = row
                        if i: 
                            break
                    else:
                        break
            
                vertexDataList[i+1][otherVertex] = row
                
            else:
                edgeCount += 1
        
        if overwroteSome:
            warnings.warn("Some edges occured multiple times. I " 
                          + ("overwrote them." if overrideExisting else 
                             ("ignored them." if ignoreNew else
                              ("used the shortest edges." if useShortest else
                               ("used the longest edges.")))))
        if loopsOccurred:
            warnings.warn("Some edges were loops. I ignored them.") 
        
        
        for vertexIndex in np.nonzero(~includedVertices)[0]:
            graph[vertexData.ID[vertexIndex]] = [vertexIndex, {}, {}]
        
        add_names(edges, ("fromID", "toID"))
        from_to = edges.ravel()
        
        if vertexData.size:
            vertices = FlexibleArray(vertexData, recArray=True)
        else:
            vertices = FlexibleArray(10, recArray=True, dtype=vertexData.dtype)
            
        edges = FlexibleArray(merge_arrays((from_to, edgeData)),
                              copy=False, recArray=True)
        
        if defaultVertexData is None:
            self.defaultVertexIndex = None
        else:
            vertices.expand(1)
            self.defaultVertexIndex = vertices.add(defaultVertexData)
        
        if defaultEdgeData is None:
            self.defaultEdgeIndex = None
        else:
            edges.expand(1)
            self.defaultEdgeIndex = edges.add(defaultEdgeData)
        
        self.vertices = vertices
        self.edges = edges
        self._edgeCount = edgeCount
    
    def add_vertex(self, vertexID, vertexData=None):
        graph = self.graph
        vertices = self.vertices
        
        if vertexID in graph:
            raise KeyError("A vertex with the specified ID already exists")
        if vertexData is None:
            vertexIndex = None
        else:
            vertexIndex = vertices.add(vertexData)
            vertices[vertexIndex].ID = vertexID
        graph[vertexID] = [vertexIndex, {}, {}]
        
        
    def remove_vertex(self, vertexID, counter=None):
        #self.increase_print_level()
        #self.prst("Removing", vertexID)
        if counter: 
            count = counter.next()
            if count: self.prst(count)
        graph = self.graph
        vertexIndex, successors, predecessors = graph.pop(vertexID)
        self._edgeCount -= len(successors) + len(predecessors)
        del self.vertices[vertexIndex]
        edges = self.edges
        for i, neighbors in enumerate((predecessors, successors)):
            for neighbor in neighbors:
                del edges[graph[neighbor][i+1].pop(vertexID)]
                
        #self.decrease_print_level()
    
    def add_edge(self, fromID, toID, edgeData=None):
        graph = self.graph
        successorDict = graph[fromID][1]
        predecessorDict = graph[toID][2]
        if toID in successorDict and fromID in predecessorDict:
            raise KeyError("The specified edge exists already")
        
        if edgeData is None:
            edgeIndex = None
        else:
            edgeIndex = self.edges.add((fromID, toID, *edgeData))
        successorDict[toID] = edgeIndex
        predecessorDict[fromID] = edgeIndex
        self._edgeCount += 1
        
    def remove_edge(self, fromID, toID):
        graph = self.graph
        successors = graph[fromID][1]
        predecessors = graph[toID][2]
        del self.edges[successors.pop(toID)]
        del predecessors[fromID]
        self._edgeCount -= 1
    
    def set_vertex_data(self, vertexID, data):
        vertexInformation = self.graph[vertexID]
        vertexIndex = vertexInformation[0]
        vertices = self.vertices
        if vertexIndex is None:
            vertexInformation[0] = vertices.add(data, ID=vertexID)
        else:
            self.vertices.array[vertexIndex] = data
    
    def set_edge_data(self, fromID, toID, data):
        graph = self.graph
        vertexFromSuccessors = graph[fromID][1]
        edges = self.edges
        edgeIndex = vertexFromSuccessors[toID]
        if edgeIndex is None:
            edgeIndex = edges.add(data, fromID=fromID, toID=toID)
            vertexFromSuccessors[toID] = edgeIndex
            graph[toID][0][fromID] = edgeIndex
        else:
            self.edges.array[edgeIndex] = data
    
    def set_default_vertex_data(self, data=None):
        if data is None:
            if self.defaultVertexIndex is None:
                return
            del self.vertices[self.defaultVertexIndex]
            self.defaultVertexIndex = None
        elif self.defaultVertexIndex is None:
            self.defaultVertexIndex = self.vertices.add(data)
        else:
            self.vertices[self.defaultVertexIndex] = data
        
    def set_default_edge_data(self, data=None):
        if data is None:
            if self.defaultEdgeIndex is None:
                return
            del self.edges[self.defaultEdgeIndex]
            self.defaultEdgeIndex = None
        if self.defaultEdgeIndex is None:
            self.defaultEdgeIndex = self.edges.add(data)
        else:
            self.vertices[self.defaultVertexIndex] = data
        
    def get_vertex_data(self, vertexID, copy=True):
        vertexInformation = self.graph[vertexID]
        vertexIndex = vertexInformation[0]
        vertices = self.vertices
        defaultVertexIndex = self.defaultVertexIndex
        
        if vertexIndex is None:
            if defaultVertexIndex is None:
                raise ValueError("Vertex has no specific properties, "
                                 + "but no default values are given.")
            if not copy:
                self.set_vertex_data(
                    vertexID, vertices.array[defaultVertexIndex]
                    )
                return vertices.array[vertexInformation[0]]
            else:
                vertexData = vertices.array[defaultVertexIndex].copy()
                vertexData.ID = vertexID
                return vertexData
        else:
            if not copy:
                return vertices.array[vertexIndex]
            else:
                return vertices.array[vertexIndex].copy()
                
    def get_edge_data(self, fromID, toID, copy=True):
        vertexFromSuccessors = self.graph[fromID][1]
        edgeIndex = vertexFromSuccessors[toID]
        defaultEdgeIndex = self.defaultEdgeIndex
        edges = self.edges
        if edgeIndex is None:
            if defaultEdgeIndex is None:
                raise ValueError("Edge has no specific properties, "
                                 + "but no default values are given.")
            if copy:
                self.set_edge_data(fromID, toID, 
                                   edges.array[defaultEdgeIndex])
                return edges.array[vertexFromSuccessors[toID]]
            else:
                edgeData = edges.array[defaultEdgeIndex].copy()
                edgeData.fromID, edgeData.toID = fromID, toID   
                return edgeData
        else:
            if copy:
                return edges.array[edgeIndex].copy()
            else:
                return edges.array[edgeIndex]
    
    def get_edge_count(self):
        return self._edgeCount
    
    def get_vertex_count(self):
        return len(self.graph)
        
    def get_successors(self, vertexID):
        return list(self.graph[vertexID][1].keys())
        
    def get_predecessors(self, vertexID):
        return list(self.graph[vertexID][2].keys())
        
    def get_neighbor_edges(self, vertexID, getSuccessors=True, copy=True):
        predSuccI = int(not getSuccessors)
        order = 1 if getSuccessors else -1
        neighborDict = self.graph[vertexID][predSuccI+1]
        edgeIndices = neighborDict.values()
        defaultEdgeIndex = self.defaultEdgeIndex
        edges = self.edges
        
        if not None in edgeIndices:
            if copy:
                return edges.array[list(edgeIndices)].copy()
            else:
                return edges.array[list(edgeIndices)]
        else:
            if defaultEdgeIndex is None:
                raise ValueError("Edge has no specific properties, "
                                 + "but no default values are given.")
            if copy:
                indexList = [(index if index is not None else defaultEdgeIndex)
                             for index in edgeIndices]
                nones = [(i, item[0]) for i, item in 
                          enumerate(neighborDict.items()) if item[1] is None]
                neighbors = self.edges.array[indexList].copy()
                fromArr = neighbors.fromID
                toArr = neighbors.toID
                for i, neighbor in nones: 
                    fromArr[i], toArr[i] = (vertexID, neighbor)[::order]     
            else:
                indexList = []
                for neighbor, edgeIndex in neighborDict.items():
                    if edgeIndex is None:
                        self.set_edge_data(*(vertexID, neighbor)[::order], 
                                           data=edges.array[defaultEdgeIndex])
                        indexList.append(neighborDict[neighbor])
                    else:
                        indexList.append(edgeIndex)
                return self.edges.array[indexList]
            
    
    def add_edge_attributes(self, names, dtypes, fillVal=None):
        self.edges.add_fields(names, dtypes, fillVal)
    
    def add_vertex_attributes(self, names, dtypes, fillVal=None):
        self.vertices.add_fields(names, dtypes, fillVal)
        
    def remove_insignificant_dead_ends(self, significanceLabel=None, 
                                       standardSignificant=False):
        
        self.prst("Removing insignificant dead ends from the graph")
        graph = self.graph
        vertices = self.vertices
        allVertexNumber = len(graph)
        
        if significanceLabel is None:
            significanceLabel = self.significanceLabel
        else:
            if type(significanceLabel) == str:
                self.significanceLabel = significanceLabel
        
        if type(significanceLabel) == str:
            significanceArr = vertices.array[significanceLabel]
        elif hasattr(significanceLabel, "__iter__"):
            significanceArr = list(significanceLabel)
        elif significanceLabel is not None:
            significanceArr = vertices.array[significanceLabel]
        else:
            significanceArr = Repeater(False)
        
        if not self.defaultVertexIndex is None:
            try:
                standardSignificant = significanceArr[self.defaultVertexIndex]
            except IndexError:
                pass
            
        queue = deque()
        
        counter = Counter(1, 1000)
        
        for vertexID in list(graph.keys()):
            queue.appendleft(vertexID)
            while len(queue):
                vertexID = queue.pop()
                try:
                    vertexIndex, successors, predecessors = graph[vertexID]
                except KeyError:
                    continue    
                if vertexIndex is None:
                    significant = standardSignificant
                else:
                    try:
                        significant = significanceArr[vertexIndex]
                    except (TypeError, IndexError):
                        significant = standardSignificant
                if not significant:
                    if not len(successors):
                        queue.extendleft(predecessors.keys())
                        self.remove_vertex(vertexID, counter)
                    elif (not len(predecessors) 
                          or (len(successors) == 1 
                              and predecessors.keys() == successors.keys())):
                        queue.extendleft(successors.keys())
                        self.remove_vertex(vertexID, counter)
        
        
        self.prst("Removed {} out of {} vertices (".format(counter.index,
                                                       allVertexNumber), end="")
        self.prst(counter.index / allVertexNumber, ")", percent=True)
    
        
class FastGraph(SeparatelySaveable, metaclass=DocMetaSuperclass):
    '''
    '''
        
    def __init__(self, flexibleGraph):
        
        SeparatelySaveable.__init__(self, extension='.rn')
        
        #can leave vertexIDs as iterator???
        rawVertexData = list(sorted(flexibleGraph.graph.items()))
        
        self.significanceLabel = flexibleGraph.significanceLabel
        
        translateVertices = {vertexID[0]:index 
                             for index, vertexID in enumerate(rawVertexData)}
        
        neighbors = np.zeros(len(rawVertexData), dtype={"names":["predecessors",
                                                             "successors"],
                                                   "formats":[object, object]})
        
        oldVertexData = flexibleGraph.vertices.array
        oldEdgeData = flexibleGraph.edges.array
        vertexData = np.zeros(len(rawVertexData), 
                              dtype=flexibleGraph.vertices.array.dtype)
        edges = np.zeros(flexibleGraph.get_edge_count(), 
                         dtype=flexibleGraph.edges.array.dtype.descr 
                                + [("fromIndex", "int"), ("toIndex", "int")])
        
        edgeFromIDArr = edges["fromID"]
        edgeToIDArr = edges["toID"]
        edgeFromIndexArr = edges["fromIndex"]
        edgeToIndexArr = edges["toIndex"]
        
        edgesViewOld = edges[list(flexibleGraph.edges.array.dtype.names)]
        
        if flexibleGraph.defaultVertexIndex is None:
            hasDefaultVertexData = False
        else:
            hasDefaultVertexData = True
            defaultVertexData = oldVertexData[flexibleGraph.defaultVertexIndex]
        
        defaultEdgeIndex = flexibleGraph.defaultEdgeIndex
        
        toBeAddedPredecessors = []
        edgeBlockIndex = 0
        for i, data in enumerate(rawVertexData):
            vertexID, vertexDataRecord = data
            if vertexDataRecord[0] is None:
                if not hasDefaultVertexData:
                    raise ValueError("Some vertices have missing information "
                                     + "and no default data are given")
                vertexData[i] = defaultVertexData
                vertexData[i].ID = vertexID
            else:
                vertexData[i] = oldVertexData[vertexDataRecord[0]]
            
            successorIndices = [translateVertices[vertexID] for vertexID in 
                                vertexDataRecord[1].keys()]
                
            neighbors[i]["successors"] = { 
                    vertexIndex:j+edgeBlockIndex
                    for j, vertexIndex in enumerate(successorIndices)
                    }
            
            neighbors[i]["predecessors"] = {}
            
            toBeAddedPredecessors.extend((item[0], (i, edgeBlockIndex+j)) 
                        for j, item in enumerate(vertexDataRecord[1].items()))
            
            indexList = [(index if index is not None else defaultEdgeIndex)
                         for index in vertexDataRecord[1].values()]
            
            nones = [(j, item[0]) for j, item in 
                      enumerate(vertexDataRecord[1].items()) if item[1] is None]
            
            try:
                edgesViewOld[edgeBlockIndex:edgeBlockIndex+len(indexList)] = \
                        oldEdgeData[indexList]
            except IndexError as e: 
                if None in indexList:
                    raise ValueError("Some edges have missing information "
                                         + "and no default data are given")
                else:
                    raise e
            
            edgeFromIndexArr[edgeBlockIndex:edgeBlockIndex+len(indexList)] = i 
            edgeToIndexArr[edgeBlockIndex:edgeBlockIndex
                                            + len(indexList)] = successorIndices
                     
            
            
            for j, neighbor in nones: 
                edgeFromIDArr[edgeBlockIndex+j] = vertexID  
                edgeToIDArr[edgeBlockIndex+j] = neighbor
            
            edgeBlockIndex += len(indexList)
                
        for vertexID, predecessorData in toBeAddedPredecessors:
            neighbors[translateVertices[vertexID]]["predecessors"
                               ][predecessorData[0]] = predecessorData[1]       
        
        
        mergeArr = np.empty(vertexData.shape[0], 
                            dtype=vertexData.dtype.descr+neighbors.dtype.descr)
         
        view = fields_view(mergeArr, vertexData.dtype.names)
        view[:] = vertexData
        view = fields_view(mergeArr, neighbors.dtype.names)
        view[:] = neighbors
        
        self.vertices = FlexibleArray(mergeArr, copy=False)
        
        self.edges = FlexibleArray(edges, copy=False)
        
        self.set_save_separately('vertices', 'edges')
        
    
    # this method is not needed.
    def make_edges_contiguous(self):
        _, newEdgeIndices = self.edges.make_contiguous()
        
        if not len(newEdgeIndices): 
            return
        
        self.edges.cut()
        vertices = self.vertices
        edgeArr = self.edges.array
        
        for indexName, oppositeIndexName, neighborName in (
                ("fromIndex", "toIndex", "successors"),
                ("toIndex", "fromIndex", "predecessors")):
        
            neighborVertices = edgeArr[newEdgeIndices][indexName]
            oppositeNeighborVertices = edgeArr[newEdgeIndices][
                                                            oppositeIndexName]
            for neighbors, oppositeNeighbor, newEdgeIndex in (
                    vertices.array[neighborName][neighborVertices],
                    oppositeNeighborVertices,
                    newEdgeIndices):
                neighbors[oppositeNeighbor] = newEdgeIndex
                
        
    def add_vertex(self, vertexData):
        return self.vertices.add(vertexData)
    
    def remove_vertex(self, vertexIndex):
        vertices = self.vertices
        edges = self.edges
        vertexData = vertices[vertexIndex]
        predSuccEssors = ("successors", "predecessors")
        for i, j in enumerate(predSuccEssors):
            for neighbor in vertexData[predSuccEssors[i-1]].keys():
                del edges[vertices[neighbor][j].pop(vertexIndex)]
        del vertices[vertexIndex]
            
    def add_edge(self, fromIndex, toIndex, edgeData):
        vertices = self.vertices
        if not vertices.exists(fromIndex, toIndex):
            raise IndexError("One of the given vertices " + 
                "{0} and {1} does not exist".format(fromIndex, toIndex))
        if toIndex in vertices[fromIndex].successors:
            raise IndexError("The specified edge exists already")
        edgeID = self.edges.add(edgeData)
        vertices[fromIndex].successors[toIndex] = edgeID
        vertices[toIndex].successors[fromIndex] = edgeID
    
    def remove_edge(self, fromIndex, toIndex):
        vertices = self.vertices
        del self.edges[vertices[fromIndex].successors.pop(toIndex)]
        del vertices[toIndex].predecessors[fromIndex]
    
    def add_edge_attributes(self, names, dtypes, fillVal=None):
        self.edges.add_fields(names, dtypes, fillVal)
    
    def add_vertex_attributes(self, names, dtypes, fillVal=None):
        self.vertices.add_fields(names, dtypes, fillVal)
        
    
class FlowPointGraph(FastGraph, HierarchichalPrinter, Lockable):
    
    def __init__(self, flexibleGraph, lengthLabel, 
                 significanceLabel=None, **printerArgs):
        
        """
        FastGraph.__init__(flexibleGraph)    
        HierarchichalPrinter.__init__(self, **printerArgs)
        """
        FastGraph.__init__(self, flexibleGraph=flexibleGraph)
        HierarchichalPrinter.__init__(self, **printerArgs)
        Lockable.__init__(self)
        
        if not significanceLabel:
            if flexibleGraph.significanceLabel:
                significanceLabel = flexibleGraph.significanceLabel
            else:
                raise ValueError("A significance label must be given either by "
                                 + "the flexible graph input or explicitely in "
                                 + " FlowPointGraph's constructor.")
        
        if (significanceLabel not in self.vertices.array.dtype.names):
            raise ValueError("There is no field " + str(significanceLabel) 
                             + " in the vertex data array.")
        
        if not significanceLabel == "significant":
            if "significant" in self.vertices.array.dtype.names:
                raise ValueError("The vertex data array must not contain a "
                                 + "field 'significant', if 'significant' is "
                                 + "not the significance label.")
            add_alias(self.vertices.array, significanceLabel, "significant")
        
        if lengthLabel not in flexibleGraph.edges.array.dtype.names:
            raise ValueError("There is no field " + str(lengthLabel) 
                             + " in the edge data array.")
        
        if not lengthLabel == "length":
            if "length" in flexibleGraph.edges.array.dtype.names:
                raise ValueError("The edge data array must not contain a field "
                                 + "'length', if 'length' is not the length "
                                 + "label.")
            add_alias(self.edges.array, lengthLabel, "length")
        
    
    def preprocessing(self, initialBound, boundFactor=3, pruneFactor=4, 
                      additionalBoundFactor=1.1, expansionBounds=None, 
                      degreeBound=5, maxEdgeLength=None):
        
        if expansionBounds is None:
            expansionBounds = Repeater(1)
        elif hasattr(expansionBounds, "__getitem__"):
            expansionBounds = iterchain(expansionBounds, 
                                        Repeater(expansionBounds[-1]))
        
        edges = self.edges
        vertices = self.vertices
        self.prst("Preprocessing the graph...")
        self.prst("initialBound:", initialBound)
        self.prst("boundFactor:", boundFactor)
        self.prst("pruneFactor:", pruneFactor)
        self.prst("additionalBoundFactor:", additionalBoundFactor)
        self.prst("expansionBounds:", expansionBounds)
        self.prst("degreeBound:", degreeBound)
        self.prst("maxEdgeLength:", maxEdgeLength)
        self.increase_print_level()
        if pruneFactor < 2:
            raise ValueError("The pruneFactor must be at least 2.")
        
        temporaryEdgesFields = ["reachBound", "considered"]
        
        edges.add_fields(temporaryEdgesFields 
                         + ["originalEdge1", "originalEdge2"],
                         ["double", "bool", "int", "int"],
                         [0, True, -1, -1])
        
        temporaryVerticesFields = ["tmp_predecessors", "tmp_successors", 
                                   "inPenalty", "outPenalty", 
                                   "inPenalty_original", "unbounded"]
        
        # the graph G'
        vertices.add_fields(["reachBound"] + temporaryVerticesFields, 
                            ["double", "object", "object", "double", "double", 
                             "double", "bool"],
                            [0, deepcopy(vertices.array["predecessors"]), 
                             deepcopy(vertices.array["successors"]), 0, 0, 0,
                             vertices.considered])
        
        if not os.name == 'posix':
            warnings.warn("Parallelization with shared memory is ony possible "
                          + "on Unix-based systems. Thus, the code will not be "
                          + "executed in parallel.")
        
        vertexArr = vertices.array[:vertices.size]
        tmp_predecessorsArr = vertexArr["tmp_predecessors"]
        tmp_successorsArr = vertexArr["tmp_successors"]
        inPenaltyArr = vertexArr["inPenalty_original"]
        outPenaltyArr = vertexArr["outPenalty"]
        totalVertexNumber = len(vertices)
        
        bound = initialBound
        changeIndex = -1
        edgeSize = edges.size
        
        while True: # break statement, if all edges are bounded
            self.prst("Computing reach bounds smaller than", bound)
            self.increase_print_level()
            
            if (not edges.changeIndex == changeIndex 
                    or not edgeSize == edges.size):
                edgeArr = edges.array[:edges.size]
                edgeConsideredArr = edges.considered[:edges.size]
                changeIndex = edges.changeIndex
                edgeSize = edges.size
            
            # we do not have to process vertices that do not have successors.
            # trees rooting there will consist of the root only.
            vertexArr["unbounded"] = np.logical_and(vertexArr["unbounded"],  
                                                    vertexArr["tmp_successors"])
            
            
            consideredVertexIndices = np.nonzero(vertexArr["unbounded"])[0]
            
            
            self._bypass_vertices(consideredVertexIndices, 
                                   bound, expansionBounds.__next__(), 
                                   degreeBound, maxEdgeLength)
            
            deletedVerticesNumber = (totalVertexNumber 
                                     - np.sum(vertexArr["unbounded"]))
            self.prst(deletedVerticesNumber, "out of", totalVertexNumber, 
                      "vertices have been removed ({:6.2%}).".format(
                                      deletedVerticesNumber/totalVertexNumber))
            
            if (not edges.changeIndex == changeIndex 
                    or not edgeSize == edges.size):
                edgeArr = edges.array[:edges.size]
                edgeConsideredArr = edges.considered[:edges.size]
                changeIndex = edges.changeIndex
                edgeSize = edges.size
            
            vertexArr["inPenalty"] = inPenaltyArr 
            
            np.fmax.at(
                vertexArr["inPenalty"], 
                edgeArr["toIndex"][edgeConsideredArr],
                edgeArr["length"][edgeConsideredArr]
                        )
            
            
            vertexArr["inPenalty"]  = np.select([vertexArr["inPenalty"] <= 
                                                        (pruneFactor-2)*bound],
                                            [vertexArr["inPenalty_original"]], 
                                            bound)
            
            
            consideredVertexIndices = np.nonzero(vertexArr["unbounded"])[0]
            
            counter = getCounter(len(consideredVertexIndices), 0.01)
            
            pruneConstant = bound*pruneFactor
            
            wrappedFunc = partial(self._get_reach_bounds_starting_from,
                                  bound=bound, 
                                  additionalBoundFactor=additionalBoundFactor,
                                  pruneConstant=pruneConstant,
                                  counter=counter)
            
            lengths = edgeArr["length"][edgeArr["considered"]]
            
            edgeArr["reachBound"][edgeArr["considered"]] = np.select(
                                    (lengths < bound,), (lengths,), (np.inf,)
                                                               )
            
            if os.name == 'posix': #and bound < 26
                
                edges.array = np.rec.array(sharedmem.copy(edges.array), 
                                           copy=False)
                with sharedmem.MapReduce(np=CPU_COUNT) as pool:
                    pool.map(wrappedFunc, consideredVertexIndices)
                    
            else:
                any(map(wrappedFunc, consideredVertexIndices))
            
            edgeArr = edges.array[:edges.size]
            edgeConsideredArr = edges.considered[:edges.size]
            changeIndex = edges.changeIndex
            edgeSize = edges.size
            self.decrease_print_level()
            
            # Delete edges with bounded reach and add penalties
            
            toBeDeleted = np.logical_and(np.logical_and(
                                edgeArr["reachBound"] < np.inf, 
                                edgeArr["considered"]
                                                        ), 
                                         edgeConsideredArr)
            
            edgeArr["considered"][toBeDeleted] = False
            
            totalEdgeNumber = len(edgeArr)
            boundedEdgesNumber = totalEdgeNumber - np.sum(edgeArr["considered"])
            self.prst(boundedEdgesNumber, "edge reaches out of", 
                      totalEdgeNumber, "are bounded (" + 
                      "{:6.2%}".format(boundedEdgesNumber/totalEdgeNumber) 
                      + ").")
            
            if not edgeArr["considered"].any(): # break statement for loop
                break
            
            for edge in edgeArr[toBeDeleted]:
                fromIndex, toIndex = edge["fromIndex"], edge["toIndex"]
                del tmp_predecessorsArr[toIndex][fromIndex]
                del tmp_successorsArr[fromIndex][toIndex]
                
                # add penalties
                if outPenaltyArr[fromIndex] < edge["reachBound"]:
                    outPenaltyArr[fromIndex] = edge["reachBound"]
                if inPenaltyArr[toIndex] < edge["reachBound"]:
                    inPenaltyArr[toIndex] = edge["reachBound"]
            
            bound *= boundFactor
        
        self._convert_edge_reaches_to_vertex_reaches()
        
        self.vertices.remove_fields(temporaryVerticesFields)
        
        self.edges.remove_fields(temporaryEdgesFields)
        
        self._sort_neighbors()
        self.edges.cut()
        
        self.decrease_print_level()
        
        
    def _get_reach_bounds_starting_from(self, rootIndex, bound, 
                                         additionalBoundFactor,
                                         pruneConstant=np.inf,
                                         counter=None):
        
        if counter:
            percentage = counter.next()
            if percentage: self.prst(percentage, percent=True)
        
        rTol = 1+1e-7
        increasedBound = bound * additionalBoundFactor
        successorArr = self.vertices.array["tmp_successors"]
        costArr = self.edges.array["length"]
        inPenaltyArr = self.vertices.array["inPenalty"]
        outPenaltyArr = self.vertices.array["outPenalty"]
        reachBoundArr = self.edges.array["reachBound"]
        
        tree = FlexibleArrayDict(4000, 
                                 dtype={"names":["vertexIndex", "depth", 
                                                 "parentMinHeight", "parent", 
                                                 "extension", "xCost", 
                                                 "successors", "visited",
                                                 "edge", "relevant", 
                                                 "improvableInnerVertex"],
                                      "formats":["int", "double", "double",
                                                 "int", "double", "double",
                                                 "object", "bool",
                                                 "int", "bool", "bool"]})
        tree.setitem_by_keywords(rootIndex, vertexIndex=rootIndex, parent=0, 
                                 successors=set(), relevant=True)
        
        queue = intquickheapdict()
        rootPenalty = inPenaltyArr[rootIndex]
        queue[rootIndex] = rootPenalty
        
        boundableEdges = set()
        relevantCount = 1
        
        # grow the tree
        
        while relevantCount:
            vertex, cost = queue.popitem()
            
            if cost-rootPenalty > pruneConstant:
                queue[vertex] = cost
                break
            
            vertexIndexInTree = tree.indexDict[vertex]
            vertexData = tree[vertex]
            vertexData["depth"] = cost
            
            # add vertex as successor of parent
            tree.array[vertexData["parent"]
                       ]["successors"].add(vertexIndexInTree)
            
            vertexData["successors"] = set()
            
            if vertexData["improvableInnerVertex"]:
                boundableEdges.add((vertexIndexInTree, vertexData["edge"]))
            
            # for the successors of vertex, vertex is the parent
            parentIndexInTree = vertexIndexInTree
            parentextension = vertexData["extension"]
            xCost = vertexData["xCost"]
            parentInPenalty = inPenaltyArr[vertex]
            parentOutPenalty = outPenaltyArr[vertex]
            parentRelevant = vertexData["relevant"]
            parentCost = cost
            depthParentSmallerInPenaltyParent = \
                                    parentCost*rTol < parentInPenalty
            
            if parentRelevant:
                relevantCount -= 1
            
            for successor, edge in successorArr[vertex].items():
                edgeCost = costArr[edge]
                newCost = parentCost + edgeCost
                
                if successor in queue:
                    successorCost = queue[successor]
                    update = successorCost > newCost * rTol #+ tolerance
                else:    # if the vertex is not in the queue, 
                                    # it either has been removed or its 
                                    # value is infinity
                    if not successor in tree:
                        update = True
                        tree.setitem_by_keywords(successor, 
                                                 vertexIndex=successor)
                    else: 
                        update = False
                
                if update:
                    # put vertex in queue / update it
                    queue[successor] = newCost
                    newVertexData = tree[successor]
                    newVertexData["parent"] = parentIndexInTree
                    
                    if not parentIndexInTree:
                        xCost = newCost-inPenaltyArr[successor]
                    
                    newVertexData["xCost"] = xCost
                    newVertexData["edge"] = edge
                    
                    # check relevance of the vertex and include it in the 
                    # queue, if necessary
                    
                    # 1.: Check whether inner vertex
                    innerVertex =  (not parentIndexInTree or
                                    ((newCost-xCost)*rTol < bound 
                                     and not depthParentSmallerInPenaltyParent))
                    
                    # calculate extension
                    # check whether improvable
                    with Locked(self):
                        edgeReachBound = reachBoundArr[edge]
                    if newCost > edgeReachBound*rTol and innerVertex:
                        newVertexData["extension"] = edgeCost
                        newVertexData["improvableInnerVertex"] = True
                    else:
                        newVertexData["extension"] = parentextension + edgeCost
                    
                    # 2. check all relevance criteria
                    if (innerVertex
                            or (parentRelevant 
                                and (parentextension+parentOutPenalty)*rTol < bound
                                and newVertexData["extension"]
                                        + outPenaltyArr[successor] 
                                                            <= increasedBound)):
                        if not newVertexData["relevant"]:
                            relevantCount += 1
                            newVertexData["relevant"] = True
                    elif newVertexData["relevant"]:
                        relevantCount -= 1
                        newVertexData["relevant"] = False
        
        treeArr = tree.array
                    
        # process leafs that have not been extended
        for leaf, depth in queue.items():
            leafIndexInTree = tree.indexDict[leaf]
            leafData = treeArr[leafIndexInTree]
            leafData["depth"] = depth
            treeArr[leafData["parent"]]["successors"].add(leafIndexInTree)
            if leafData["improvableInnerVertex"]:
                boundableEdges.add((leafIndexInTree, leafData["edge"]))
        
        
        # compute height of elements in tree. Thereby, assign to each vertex
        # the height of its parent, if it would be reached through this vertex
        stack = [0]
        
        while stack:
            vertexData = treeArr[stack[-1]]
            if not vertexData["visited"] and vertexData["successors"]:
                vertexData["visited"] = True
                stack.extend(vertexData["successors"])
            else:
                stack.pop()
                parentData = treeArr[vertexData["parent"]]
                newHeight = (max(vertexData["parentMinHeight"], 
                                 outPenaltyArr[vertexData["vertexIndex"]]) 
                             + costArr[vertexData["edge"]])
                
                # change the own height to the height of the parent, if it 
                # would be reached through the vertex
                vertexData["parentMinHeight"] = newHeight
                
                # change the real height of the parent
                # (now parentMinHeight contains the real height, not the 
                #  parent's - this will be changed later - see above)
                if parentData["parentMinHeight"]*rTol < newHeight:
                    parentData["parentMinHeight"] = newHeight
        
        depthArray = tree.array["depth"]
        heightArray = tree.array["parentMinHeight"]
                
        for successorIndexInTree, edge in boundableEdges:
            newReachBound = min(depthArray[successorIndexInTree],
                                heightArray[successorIndexInTree])
            if newReachBound*rTol >= bound:
                newReachBound = np.inf
            with Locked(self):
                if newReachBound > reachBoundArr[edge]:
                    reachBoundArr[edge] = newReachBound
    
    def _bypass_vertices(self, vertexIndices, reachBound, 
                          expansionBound, degreeBound, maxEdgeLength=None):
        
        self.prst("Determining vertex weights.")
        self.increase_print_level()
        counter = ParallelCounter(len(vertexIndices), 0.01)
        
        wrappedFunc = partial(self._determine_vertex_weight, 
                              reachBound=reachBound,
                              expansionBound=expansionBound, 
                              degreeBound=degreeBound,
                              maxEdgeLength=maxEdgeLength,
                              counter=counter)
        
        if os.name == 'posix':
            with sharedmem.MapReduce(np=CPU_COUNT) as pool:
                vertexWeights = np.array(pool.map(wrappedFunc, vertexIndices))
        else:
            vertexWeights = np.array(tuple(map(wrappedFunc, vertexIndices)))
        
        considered = np.logical_not(np.isnan(vertexWeights))
        
        vertexWeights = intquickheapdict(zip(vertexIndices[considered], 
                                          vertexWeights[considered]), 
                                         )
        
        self.decrease_print_level()
        self.prst("Bypassing vertices")       
        self.increase_print_level()
        
        counter = Counter(1, 1000)
        dictIterator = DictIterator(vertexWeights, np.inf)
        any(map(partial(self._bypass_vertex, 
                        reachBound=reachBound,
                        expansionBound=expansionBound, 
                        degreeBound=degreeBound, 
                        vertexWeights=vertexWeights,
                        maxEdgeLength=maxEdgeLength,
                        counter=counter), 
                dictIterator))
        self.prst("Bypassed", dictIterator.count, "out of", len(vertexIndices),
                  "vertices ({:6.2%}).".format(dictIterator.count/
                                               len(vertexIndices)))
        self.decrease_print_level()
        
    
    def _determine_vertex_weight(self, vertexIndex, reachBound, expansionBound, 
                                 degreeBound, maxEdgeLength=None, 
                                 vertexWeights=None, counter=None):
        
        if vertexWeights is not None and vertexIndex not in vertexWeights:
            return
        
        if not counter is None: 
            percentage = counter.next()
            if percentage: self.prst(percentage, percent=True)
            
        vertexData = self.vertices.array[vertexIndex]
        successors = vertexData["tmp_successors"]
        predecessors = vertexData["tmp_predecessors"]
        successorArr = self.vertices.array["successors"]
        costBound = reachBound / 2
        
        # check simple absolute bounds
        
        inDegree = len(predecessors)
        outDegree = len(successors)
        
        if not inDegree + outDegree:
            self.vertices.array["unbounded"][vertexIndex] = False
            if vertexWeights:
                del vertexWeights[vertexIndex]
                return
            else:
                return np.nan
            
        # 1. Degree bound  
        if inDegree >= degreeBound or outDegree >= degreeBound:
            # the degree can decrease within one bypassing run, if neighbors
            # get deleted
            if vertexWeights:
                vertexWeights[vertexIndex] = np.inf
                return
            else:
                return np.inf
        
        # determining expansion (# of new edges (including remaining old edges)
        #                             / # of old edges)
        if vertexData["significant"]:
            replaceEdges = False
        else:
            replaceEdges = (inDegree == len(vertexData["predecessors"])
                            and
                            outDegree == len(vertexData["successors"]))
        
        lengthArr = self.edges.array["length"]
        inPenalty = vertexData["inPenalty_original"]
        outPenalty = vertexData["outPenalty"]
        
        if maxEdgeLength and successors and predecessors:
            successorEdgeList = list(successors.values())
            predecessorEdgeList = list(predecessors.values())
            successorList = list(successors.keys())
            predecessorList = list(predecessors.keys())
            maxSuc1 = np.argmax(lengthArr[successorEdgeList])
            maxSucVal1 = lengthArr[successorEdgeList[maxSuc1]]
            maxPred2 = np.argmax(lengthArr[predecessorEdgeList])
            maxPredVal2 = lengthArr[predecessorEdgeList[maxPred2]]
            try:
                del predecessorEdgeList[predecessorList.index(successorList[maxSuc1])]
            except ValueError:
                pass
            try:
                del successorEdgeList[successorList.index(predecessorList[maxPred2])]
            except ValueError:
                pass
            
            maxLen = -1
            if predecessorEdgeList:
                maxLen = maxSucVal1 + np.max(lengthArr[predecessorEdgeList])
            if successorEdgeList:
                maxLen = max(maxPredVal2 + np.max(lengthArr[successorEdgeList]), maxLen)
                
            if maxLen > maxEdgeLength:
                if vertexWeights:
                    vertexWeights[vertexIndex] = np.inf
                    return
                else:
                    return np.inf
                
        if expansionBound is None:
            stopper = np.inf
        else: 
            # if we cannot replace edges, we are not allowed to create as 
            # many new edges 
            stopper = min((outDegree+inDegree) * 
                          (expansionBound-(not replaceEdges)), 
                          2*degreeBound - inDegree - outDegree)
        
        
        # Could use list comprehension in order to improve speed...
        newEdgeCount = 0
        cost = 0
        for successorItem, predecessorItem in iterproduct(successors.items(), 
                                                          predecessors.items()):
            successor, successorEdge = successorItem
            predecessor, predecessorEdge = predecessorItem
            if successor == predecessor: continue
            
            newEdgeCount += successor not in successorArr[predecessor]
            cost = max(lengthArr[successorEdge] + inPenalty,
                       lengthArr[predecessorEdge] + outPenalty,
                       lengthArr[successorEdge] + lengthArr[predecessorEdge],
                       cost)
            
            # 2. Expansion bound
            if newEdgeCount > stopper:
                if vertexWeights:
                    # the expansion can change within this bypassing round
                    # => we do not delete the vertex
                    vertexWeights[vertexIndex] = np.inf
                    return
                else:
                    return np.inf
            
            # 3. Cost bound
            if cost > costBound:
                if vertexWeights:
                    # the vertex cost can only decrease within this round if the
                    # triangle inequality is not satisfied. Though this is possile,
                    # it will not happen often and reexamining the vertex again
                    # would not be worth it => we delete the vertex
                    vertexWeights[vertexIndex] = np.inf
                    return
                else:
                    return np.inf
                    
        
        expansion = newEdgeCount / (outDegree + inDegree) + (not replaceEdges)
        if vertexWeights:
            vertexWeights[vertexIndex] = expansion * cost
        else:
            return expansion * cost
    
    def _bypass_vertex(self, vertexIndex, reachBound, expansionBound, 
                        degreeBound, vertexWeights, maxEdgeLength=None, counter=None):
        # note: if the vertex does not belong to any shortest path that is 
        #       not included in G' anymore, we can completely remove it from
        #       the graph (i.e. replace edges instead of only adding ne ones)
        
        if counter: 
            count = counter.next()
            if count: self.prst(count)
            
        edges = self.edges
        vertices = self.vertices
        edgeArr = edges.array
        vertexArr = vertices.array
        successorArr = vertexArr["successors"]
        predecessorArr = vertexArr["predecessors"]
        tmp_successorArr = vertexArr["tmp_successors"]
        tmp_predecessorArr = vertexArr["tmp_predecessors"]
        inPenaltyArr = vertexArr["inPenalty_original"]
        outPenaltyArr = vertexArr["outPenalty"]
        successors = tmp_successorArr[vertexIndex]
        predecessors = tmp_predecessorArr[vertexIndex]
        lengthArr = edgeArr["length"]
        consideredArr = edgeArr["considered"]
        reachBoundArr = edgeArr["reachBound"]
        inspectionArr = edgeArr["inspection"]
        previousNeighbors = set(iterchain(successors.keys(), 
                                          predecessors.keys()))
        
        vertexArr["unbounded"][vertexIndex] = False
        
        if vertexArr["significant"][vertexIndex]:
            replaceEdges = False
        else:
            replaceEdges = (len(predecessors) == len(predecessorArr[vertexIndex])
                            and
                            len(successors) == len(successorArr[vertexIndex]))
        
        if not replaceEdges:
            inPenalty = inPenaltyArr[vertexIndex]
            outPenalty = outPenaltyArr[vertexIndex]
        
        changeIndex = edges.changeIndex
        for successorItem, predecessorItem in iterproduct(successors.items(), 
                                                          predecessors.items()):
            successor, successorEdge = successorItem
            predecessor, predecessorEdge = predecessorItem
            
            # introduce new edges
            if not successor == predecessor:
                newLength = lengthArr[successorEdge] + lengthArr[predecessorEdge]
                
                if newLength > maxEdgeLength:
                    if vertexIndex in vertexWeights:
                        print("vertexWeights[vertexIndex]", vertexWeights[vertexIndex])
                    else:
                        print("vertex not in VertexWeights")
                    print("Too long edge inserted", vertexArr[vertexIndex]["ID"], newLength,
                          self._determine_vertex_weight(vertexIndex,
                        reachBound=reachBound, 
                        expansionBound=expansionBound, 
                        degreeBound=degreeBound,
                        maxEdgeLength=maxEdgeLength))
                
                successorsOfPredecessor = successorArr[predecessor]
                if inspectionArr[successorEdge]:
                    if inspectionArr[predecessorEdge]:
                        newInspection = inspectionArr[successorEdge
                                                      ].union(inspectionArr[
                                                          predecessorEdge])
                    else:
                        newInspection = inspectionArr[successorEdge]
                elif inspectionArr[predecessorEdge]: 
                    newInspection = inspectionArr[predecessorEdge]
                else:
                    newInspection = None
                if successor not in successorsOfPredecessor:
                    newIndex = edges.add_by_keywords(fromIndex=predecessor, 
                                                  toIndex=successor,
                                                  length=newLength,
                                                  considered=True,
                                                  originalEdge1=predecessorEdge,
                                                  originalEdge2=successorEdge,
                                                  inspection=newInspection)
                    successorArr[predecessor][successor] = newIndex
                    predecessorArr[successor][predecessor] = newIndex
                    tmp_successorArr[predecessor][successor] = newIndex
                    tmp_predecessorArr[successor][predecessor] = newIndex
                elif lengthArr[successorsOfPredecessor[successor]] > newLength:
                    newIndex = successorsOfPredecessor[successor]
                    edges.setitem_by_keywords(newIndex, fromIndex=predecessor, 
                                           toIndex=successor,
                                           length=newLength, considered=True,
                                           originalEdge1=predecessorEdge,
                                           originalEdge2=successorEdge,
                                           inspection=newInspection)
                    if successor not in tmp_successorArr[predecessor]:
                        tmp_successorArr[predecessor][successor] = newIndex
                        tmp_predecessorArr[successor][predecessor] = newIndex
                        
        
            if not changeIndex == edges.changeIndex:
                edgeArr = edges.array
                inspectionArr = edgeArr["inspection"]
                lengthArr = edgeArr["length"]
                consideredArr = edgeArr["considered"]
                reachBoundArr = edgeArr["reachBound"]
                changeIndex = edges.changeIndex
            
        # delete old edges     
        for items, neighborArr in ((successors.items(), tmp_predecessorArr),
                                   (predecessors.items(), tmp_successorArr)):   
            for neighbor, edge in items:
                #neighborArr[neighbor].pop(vertexIndex, None)
                del neighborArr[neighbor][vertexIndex]
                consideredArr[edge] = False
                
        if replaceEdges:
            for neighbors, neighborArr in ((successors.keys(), predecessorArr),
                                           (predecessors.keys(), successorArr)):   
                for neighbor in neighbors:
                    #neighborArr[neighbor].pop(vertexIndex, None)
                    del neighborArr[neighbor][vertexIndex]
            successorArr[vertexIndex].clear()
            predecessorArr[vertexIndex].clear()
            
            # this step is is not necessary, but good style 
            # (setting the reach of deleted edges to 0)
            for neighborEdges in (successors.values(), predecessors.values()):
                reachBoundArr[list(neighborEdges)] = 0
        else:
            for neighborItems, penalty, penaltyArr in ((successors.items(), 
                                                        inPenalty, 
                                                        inPenaltyArr),
                                                       (predecessors.items(), 
                                                        outPenalty,
                                                        outPenaltyArr)):   
                for item in neighborItems:
                    vertex, edge = item
                    deletedPathLength = lengthArr[edge] + penalty
                    
                    # set reach of deleted edge
                    reachBoundArr[edge] = deletedPathLength
                    
                    # update inpenalty of the neighbor
                    if deletedPathLength > penaltyArr[vertex]: 
                        penaltyArr[vertex] = deletedPathLength 
                    
        
        successors.clear()
        predecessors.clear()
        
        any(map(partial(self._determine_vertex_weight,
                        reachBound=reachBound, 
                        expansionBound=expansionBound, 
                        degreeBound=degreeBound,
                        maxEdgeLength=maxEdgeLength, 
                        vertexWeights=vertexWeights), 
                previousNeighbors))
                   
    def _convert_edge_reaches_to_vertex_reaches(self):
        
        edgeReachArr = self.edges.array["reachBound"]
        
        for vertex in self.vertices:
            
            if (not vertex["predecessors"]
                    or not vertex["successors"]):
                vertex["reachBound"] = 0
                continue
            
            predecessorEdgeReaches = edgeReachArr[
                                             list(vertex["predecessors"].values())
                                                  ]
            successorEdgeReaches = edgeReachArr[
                                            list(vertex["successors"].values())
                                                ]
            
            predecessorList = list(vertex["predecessors"].keys())
            successorList = list(vertex["successors"].keys())
            
            predArgMax1 = np.argmax(predecessorEdgeReaches)
            predMax1 = predecessorEdgeReaches[predArgMax1]
            
            try: 
                predArgMax1Succ = \
                            successorList.index(predecessorList[predArgMax1])
            except ValueError: # if maximal predecessor is not successor
                vertex["reachBound"] = min(predMax1, np.max(successorEdgeReaches))
                continue
            
            succArgMax2 = np.argmax(successorEdgeReaches)
            succMax2 = successorEdgeReaches[succArgMax2]
            
            try: 
                succArgMax1Pred = \
                            predecessorList.index(successorList[succArgMax2])
            except ValueError: # if maximal successor is not predecessor 
                vertex["reachBound"] = min(succMax2, 
                                        np.max(predecessorEdgeReaches))
                continue
            
            try:
                succMax1 = max(reach for i, reach in 
                               enumerate(successorEdgeReaches) 
                               if not i == predArgMax1Succ) 
            except ValueError:
                succMax1 = 0
            
            try:
                predMax2 = max(reach for i, reach in 
                               enumerate(predecessorEdgeReaches) 
                               if not i == succArgMax1Pred) 
            except ValueError:
                predMax2 = 0
            
            vertex["reachBound"] = max(min(predMax1, succMax1), 
                                       min(succMax2, predMax2))
    
    def _sort_neighbors(self):
        
        self.prst("Sorting the neighbor dictionaries with respect to reach",
                  "bounds")
        vertexArr = self.vertices.get_array()
        reachArr = self.vertices.array["reachBound"]
        counter = ParallelCounter(2*len(vertexArr), 0.01)
        
        def getOrderedDicts(neighbors):
            percentage = counter.next()
            if percentage:
                self.prst(percentage, percent=True)
            keys = np.array(tuple(neighbors.keys()), dtype=int)
            values = np.array(tuple(neighbors.values()), dtype=int)
            order = np.argsort(reachArr[keys])[::-1]
            return FixedOrderedIntDict(keys[order], values[order],
                                       neighbors, copy=False, check=False)
        
        for name in "predecessors", "successors":
            if os.name == 'posix':
                with sharedmem.MapReduce(np=CPU_COUNT) as pool:
                    vertexArr[name] = pool.map(getOrderedDicts, vertexArr[name])
            else:
                vertexArr[name] = tuple(map(getOrderedDicts, vertexArr[name]))
                
            """
            with Pool(CPU_COUNT) as pool:
                vertexArr[name] = pool.map(getOrderedDicts, vertexArr[name])
            """
    
    def find_shortest_path(self, fromIndex, toIndex, getPath=False, 
                           initSize=2000):
        vertexArr = self.vertices.array
        reachArr = vertexArr["reachBound"]
        successorArr = vertexArr["successors"]
        predecessorArr = vertexArr["predecessors"]
        edgeArr = self.edges.array
        lengthArr = edgeArr["length"]
        
        forwardData = FlexibleArrayDict(initSize, 
                                        dtype={"names":["cost", "parent", 
                                                        "edge"],
                                               "formats":[float, int, int]})
        forwardData[fromIndex] = (0, -1, -1)
        forwardDataDict = forwardData.indexDict
        
        backwardData = FlexibleArrayDict(initSize, 
                                         dtype={"names":["cost", "parent", 
                                                         "edge"],
                                                "formats":[float, int, int]})
        backwardData[toIndex] = (0, -1, -1)
        backwardDataDict = backwardData.indexDict
        
        forwardQueue = intquickheapdict(((fromIndex, 0),), initSize)
        backwardQueue = intquickheapdict(((toIndex, 0),), initSize)
        
        forwardVertex, forwardCost = fromIndex, 0
        backwardVertex, backwardCost = toIndex, 0
        bestLength = np.inf
        
        while bestLength > forwardCost + backwardCost:
            if forwardCost <= backwardCost:
                thisQueue = forwardQueue
                oppositeQueue = backwardQueue
                thisVertex = forwardVertex
                thisCost = forwardCost
                oppositeCost = backwardCost
                neighborArr = successorArr
                thisDataDict = forwardDataDict
                thisData = forwardData
                oppositeDataDict = backwardDataDict
                oppositeData = backwardData
            else:
                thisQueue = backwardQueue
                oppositeQueue = forwardQueue
                thisVertex = backwardVertex
                thisCost = backwardCost
                oppositeCost = forwardCost
                neighborArr = predecessorArr
                thisDataDict = backwardDataDict
                thisData = backwardData
                oppositeDataDict = forwardDataDict
                oppositeData = forwardData
                
            
            # delete item from queue
            thisQueue.popitem()
            
            # prune, if necessary (This step is necessary, since
            # early pruning is weakened in order to allow for the fancy
            # termination criterion
            if not reachArr[thisVertex] < thisCost:
                    
                # check whether the vertex has been labeled from the opposite 
                # side
                reverseCost = oppositeQueue.get(thisVertex, -1.)
                if reverseCost >= 0:                             # if yes
                    totalLength = thisCost + reverseCost
                    # update best path if necessary
                    if totalLength < bestLength:
                        bestLength = totalLength
                        bestPathMiddleEdge = oppositeData[thisVertex]["edge"]
                
                # set the vertex cost           
                thisData[thisVertex]["cost"] = thisCost
                
                # process successors
                for neighbor, edge in neighborArr[thisVertex].items():
                    
                    # early pruning
                    reach = reachArr[neighbor]
                    length = lengthArr[edge]
                    newCost = thisCost + length
                    if reach < oppositeCost:
                        if reach < thisCost:
                            break
                        elif reach < newCost:
                            continue
                    
                    # if not pruned
                    neighborCost = thisQueue.get(neighbor, -1.)
                    
                    if neighborCost >= 0:   # if neighbor is in the queue
                        if neighborCost > newCost:
                            neighborData = thisData[neighbor]
                            neighborData["parent"] = thisVertex
                            neighborData["edge"] = edge
                            update = True
                        else:
                            update = False
                    else:
                        #check whether neighbor already scanned
                        if not neighbor in thisDataDict:
                            thisData[neighbor] = (np.nan, thisVertex, edge)
                            update = True
                        else:
                            update = False
                        
                    if update:
                        thisQueue[neighbor] = thisCost + length
                        
                        # check whether neighbor has been scanned from the
                        # opposite direction and update the best path 
                        # if necessary
                        oppositeIntIndex = oppositeDataDict.get(neighbor, -1.)
                        if oppositeIntIndex >= 0:
                            reverseCost = oppositeData.array["cost"
                                                             ][oppositeIntIndex]
                            if not np.isnan(reverseCost):
                                totalLength = (thisCost + length 
                                               + reverseCost)
                                if totalLength < bestLength:
                                    bestLength = totalLength
                                    bestPathMiddleEdge = edge
            
            if bestLength > forwardCost + backwardCost:        
                try:
                    if forwardCost <= backwardCost:
                        forwardVertex, forwardCost = thisQueue.peekitem()
                    else:
                        backwardVertex, backwardCost = thisQueue.peekitem()
                except IndexError:
                    warnings.warn("Vertices {} and {} are disconnected.".format(
                        self.vertices[fromIndex]["ID"], 
                        self.vertices[toIndex]["ID"]))
                    break
        
        if getPath:
            fromIndexArr = edgeArr["fromIndex"]
            toIndexArr = edgeArr["toIndex"]
            originalEdgeArr = edgeArr[["originalEdge2", "originalEdge1"]]
            isShortcutArr = edgeArr["originalEdge1"] >= 0
            
            def expandEdge(edgeIndex):
                result = []
                stack = [edgeIndex]
                while stack:
                    edge = stack.pop()
                    if isShortcutArr[edge]:
                        stack.extend(originalEdgeArr[edge])
                    else:
                        result.append(edge)
                return result
            
            pathEdgeIndices = deque(expandEdge(bestPathMiddleEdge))
            while not fromIndexArr[pathEdgeIndices[0]] == fromIndex:
                pathEdgeIndices.extendleft(expandEdge(forwardData[fromIndexArr[
                                           pathEdgeIndices[0]]]["edge"])[::-1])
            while not toIndexArr[pathEdgeIndices[-1]] == toIndex:
                pathEdgeIndices.extend(expandEdge(backwardData[toIndexArr[
                                              pathEdgeIndices[-1]]]["edge"]))
            
            return bestLength, edgeArr[list(pathEdgeIndices)]
        
        return bestLength
    
    
    
    def find_shortest_distance_array(self, fromIndices, toIndices):
        
        self.prst("Computing shortest distance array")
        self.increase_print_level()
        sinkNumber = len(toIndices)
        sourceNumber = len(fromIndices)
        dists = np.empty((sourceNumber, sinkNumber))
        sourceSinkCombinations = np.array(list(iterproduct(fromIndices, 
                                                           toIndices)))
        
        combinationNumber = sourceSinkCombinations.shape[0]
        chunk_number = 5
        min_chunk_size = 1
        cpu_count = max(min(CPU_COUNT, len(sourceSinkCombinations) // 10), 1)
        chunksize = max(min_chunk_size, len(sourceSinkCombinations)//
                        (cpu_count*chunk_number))
        
        const_args = (self.vertices.array, self.edges.array)
        
        printCounter = Counter(combinationNumber, 0.01)
        
        with ProcessPoolExecutor_ext(cpu_count, const_args) as pool:
                mapObj = pool.map(
                        find_shortest_distance,
                        sourceSinkCombinations[:,0],
                        sourceSinkCombinations[:,1],
                        chunksize=chunksize
                        )
                                                                 
                for i, distance in enumerate(mapObj):
                    percentage = printCounter.next()
                    if percentage is not None:
                        self.prst(percentage, percent=True)
                    dists[i // sinkNumber, i % sinkNumber] = distance
        self.decrease_print_level()
        return dists
        
    def find_alternative_paths(self, *args, **kwargs):
        warnings.warn("The function `find_alternative_paths` has been renamed"
                      "to `find_locally_optimal_paths.`", DeprecationWarning) 
        return self.find_locally_optimal_paths(*args, **kwargs)
    
    def find_locally_optimal_paths(self, fromIndices, toIndices, 
                               shortestDistances=None,
                               stretchConstant=1.5,         # beta
                               localOptimalityConstant=.2,  # alpha
                               acceptionFactor=0.9,
                               rejectionFactor=1.1,
                               testing=False
                               ):           
        
        self.prst("Finding alternative paths (parallelly)")
        self.increase_print_level()
        self.prst("Stretch Constant:", stretchConstant)
        self.prst("local Optimality Constant:", localOptimalityConstant)
        self.prst("acception factor:", acceptionFactor)
        self.prst("rejection factor:", rejectionFactor)
        self.prst("|O|, |D| = ", len(fromIndices), len(toIndices))
        if testing:
            self.prst("Testing:", testing)
        
        if hasattr(testing, "__contains__"):
            testing_args = testing
        else:
            testing_args = {}
        
        testResults = {"time":{}, "result":{}}
        
        
        startTime = startTimeTot = time.time()
        
        # compute shortest distances @@@
        if shortestDistances is None:
            dists = self.find_shortest_distance_array(fromIndices, toIndices)
            # treat disconnected sources and sinks well
            sourcesConsidered = np.min(dists, 1) < np.inf
            sinksConsidered = np.min(dists, 0) < np.inf
            dists = dists[sourcesConsidered][:,sinksConsidered]
            fromIndices = fromIndices[sourcesConsidered]
            toIndices = toIndices[sinksConsidered]
            if (dists == np.inf).any():
                warnings.warn("Some sources and sinks are disconnected though " +
                              "no source is separated from all sinks and " +
                              "vice versa. That is, the graph has separate " +
                              "subgraphs.")
        else:
            dists = shortestDistances
        
        endTime = time.time()
        testResults["time"]["shortest path"] = endTime-startTime
        self.prst("Searching shortest paths took {} seconds.".format(round(endTime-startTime, 2)))
        startTime = endTime
        
        max_source_dists = np.max(dists, 1)
        max_sink_dists = np.max(dists, 0)
        min_source_dists = np.min(dists, 1)
        min_sink_dists = np.min(dists, 0)
        
        sourceNumber = len(fromIndices)
        sinkNumber = len(toIndices)

        startIndices = np.concatenate((fromIndices, toIndices))
        
        manager = Manager()
        closestSourceQueue = manager.Queue()
        
        cpu_count = min(CPU_COUNT, max(sinkNumber//10, 1))
        const_args = [self.vertices.array, self.edges.array["length"], 
                      self.edges.array["inspection"].astype(bool),
                      stretchConstant, localOptimalityConstant, 
                      closestSourceQueue]
        
        taskLength = len(startIndices)
        self.prst("Labelling vertices.")
        self.increase_print_level()
        
        printCounter = Counter(taskLength, 0.005)
        
        labelData = []
        
        closestSourceDists = np.full(self.vertices.size, np.inf)
        edgesVisitedSources = defaultdict(set)
        edgesVisitedSinks = defaultdict(set)
        
        # task must be split to have the correct lower distence bounds
        with ProcessPoolExecutor_ext(cpu_count, const_args) as pool:
            mapObj = pool.map(FlowPointGraph._grow_bounded_tree,
                              fromIndices, Repeater(True), min_source_dists,
                              max_source_dists, 
                              Repeater("tree_bound" in testing_args),
                              Repeater("pruning_bound" in testing_args),
                              chunksize=1)
            decreaseThread = threading.Thread(
                        target=FlowPointGraph._decrease_closest_source_distance, 
                        args=(closestSourceQueue, closestSourceDists)
                        )
            decreaseThread.start()
            
            for i, item in enumerate(mapObj):
                percentageDone = printCounter.next()
                if percentageDone:
                    self.prst(percentageDone, percent=True)
                tree, edgesVisited = item
                for s in edgesVisited:
                    edgesVisitedSources[s].add(i)
                labelData.append(tree)
            
            closestSourceQueue.put(None)
            decreaseThread.join()
        
        self.prst("Processed sources. Now processing the sinks.")
        const_args[-1] = closestSourceDists
        with ProcessPoolExecutor_ext(cpu_count, const_args) as pool:
            mapObj = pool.map(FlowPointGraph._grow_bounded_tree,
                              toIndices, 
                              Repeater(False), min_sink_dists, max_sink_dists, 
                              Repeater("tree_bound" in testing_args),
                              Repeater("pruning_bound" in testing_args),
                              Repeater("pruning_bound_extended" in testing_args),
                              chunksize=5)
            for i, item in enumerate(mapObj):
                percentageDone = printCounter.next()
                if percentageDone:
                    self.prst(percentageDone, percent=True)
                tree, edgesVisited = item
                for s in edgesVisited:
                    if s in edgesVisitedSources:
                        edgesVisitedSinks[s].add(i)
                labelData.append(tree)
                
        self.prst("Vertex labelling done.")
        endTime = time.time()
        testResults["time"]["labelling"] = endTime-startTime
        self.prst("Labelling took {} seconds.".format(round(endTime-startTime, 2)))
        startTime = endTime
        self.decrease_print_level()
        
        
        
        consideredEdges = np.array(tuple(edgesVisitedSinks.keys()))
        
        
        if "reject_identical" in testing_args or "find_plateaus" in testing_args:
            plateauPeakEdges = np.array(list(set(consideredEdges).intersection(
                    edgesVisitedSources.keys())))
        else:
            plateauPeakEdges = self._find_plateau_peaks(consideredEdges, 
                                                        edgesVisitedSources,
                                                        edgesVisitedSinks)
        
        endTime = time.time()
        testResults["result"]["number plateau peaks"] = len(plateauPeakEdges)
        testResults["result"]["number labelled edges"] = len(consideredEdges)
        testResults["time"]["plateau peaks"] = endTime-startTime
        self.prst("Identifying plateau peaks took {} seconds.".format(round(endTime-startTime, 2)))
        startTime = endTime
        
        lengths = self.edges.array["length"][plateauPeakEdges]
        #print("Mean length", np.mean(lengths))
        #print("50th, 60th, 70th, 80th, 90th, 95th  percentile", np.percentile(lengths, [50, 60, 70, 80, 90, 95]))
        
        self.prst("Noting the distances from the significant edges",
                  "to all sources and sinks")
        self.increase_print_level()
        
        
        findPairProduct = lambda i: len(edgesVisitedSources[i])*len(edgesVisitedSinks[i])
        order = np.argsort(tuple(map(findPairProduct, 
                                     plateauPeakEdges)))
        
        
        plateauPeakEdges = plateauPeakEdges[order[::-1]]
        
        
        taskLength = len(plateauPeakEdges)
        const_args = [labelData]
        
        sourceDistances = np.empty((taskLength, sourceNumber))
        sinkDistances = np.empty((taskLength, sinkNumber))
        printCounter = Counter(taskLength, 0.005)
        
        # we only need to consider the toIndex of the edges,
        # because the outer vertex might have been pruned in the 
        # backwards labelling process and there the outer vertex is the
        # fromVertex
        
        with ProcessPoolExecutor_ext(None, const_args) as pool:
            mapObj = pool.map(FlowPointGraph._get_edge_source_sink_distances,
                              [edgesVisitedSources[e] for e in plateauPeakEdges],
                              [edgesVisitedSinks[e] for e in plateauPeakEdges],
                              self.edges.array["toIndex"][plateauPeakEdges],
                              repeat(sourceNumber), repeat(sinkNumber),
                              tasklength=taskLength)
            for i, distanceTuple in enumerate(mapObj):
                percentageDone = printCounter.next()
                if percentageDone:
                    self.prst(percentageDone, percent=True)
                sourceDistances[i] = distanceTuple[0]
                sinkDistances[i] = distanceTuple[1]
        
        endTime = time.time()
        testResults["time"]["source sink distance preparation"] = endTime-startTime
        self.prst("Noting distances to sources and sinks took {} seconds.".format(round(endTime-startTime, 2)))
        startTime = endTime
        self.decrease_print_level() 
        
        self.prst("Determining unique candidates per plateau.")
        self.increase_print_level()
        
        taskLength = sourceNumber*sinkNumber
        printCounter = Counter(taskLength, 0.005)
        
        const_args = (dists, sourceDistances, sinkDistances, 
                      stretchConstant, 
                      self.edges.array["toIndex"][plateauPeakEdges])
        
        dtype = [("source", np.long), ("sink", np.long), ("distance", float)]
        pairData = defaultdict(lambda: FlexibleArray(500, dtype=dtype))
        
        pairVertexCount = 0
        with ProcessPoolExecutor_ext(None, const_args) as pool:
            mapObj = pool.map(FlowPointGraph._find_vertexCandidates,
                iterproduct(range(sourceNumber), range(sinkNumber)),
                repeat("reject_identical" not in testing_args), tasklength=taskLength)
            for pair, res in zip(iterproduct(range(sourceNumber), 
                                             range(sinkNumber)), mapObj):
                pairViaVertices, lengths = res
                percentageDone = printCounter.next()
                if percentageDone:
                    self.prst(percentageDone, percent=True)
                for v, l in zip(pairViaVertices, lengths):
                    pairData[v].add_tuple((*pair, l)) 
                pairVertexCount += len(pairViaVertices)
        
        self.prst("{} via candidates are left. ({} vertices per pair)".format(
                len(pairData), pairVertexCount/sourceNumber/sinkNumber))
        
        self.decrease_print_level()
        
        endTime = time.time()
        testResults["time"]["length and uniqueness"] = endTime-startTime
        testResults["result"]["number unique candidates"] = len(pairData)
        testResults["result"]["number unique candidates pair"] = pairVertexCount
        self.prst("Determining unique candidates per plateau took {} seconds.".format(round(endTime-startTime, 2)))
        startTime = endTime
        
        self.prst("Checking the admissibility of the candidate vertices.")
        taskLength = len(pairData)
        
        
        self.increase_print_level()
        
        
        printCounter = Counter(taskLength, 0.005)
        chunksize = 40
        if taskLength > 100:
            cpu_count = None
        else:
            cpu_count = 1
        
        # We cannot use only the vertices, because if a vertex appears once 
        # with sources 1, 2, 3 and sinks A, B  and once with 2, B, C, then
        # merging sinks and sources would leave us with 1, 2, 3 times A, B, C.
        # This could become a very disadvantageous increase in combinations.
        # Furthermore, we only want to consider sources and sinks that reach
        # the vertex from different directions. 
        # Therefore, we can merge the vertices only, if we find a smart way
        # to keep the number of considered pairs small.
        
        countVia = 0 #debug only
        countNotLO = 0 #debug only
        disorder = np.arange(taskLength, dtype=int)
        np.random.shuffle(disorder)
        viaVertices = np.zeros((sourceNumber, sinkNumber), dtype=object)
        
        if testing:
            pathLengths = []
        else:
            pathLengths = np.zeros((sourceNumber, sinkNumber), dtype=object)
        viaVertexCount = np.zeros((sourceNumber, sinkNumber), dtype=int)
        inspectedRoutes = defaultdict(lambda: defaultdict(list))
        inspectionArr = self.edges.array["inspection"]
        stationCombinations = defaultdict(list)
        
        viaCandidates = np.array(list(pairData.keys()))
        viaData = np.array([arr.get_array() for arr in pairData.values()])
        
        const_args = [self.vertices.array, self.edges.array, dists, 
                      labelData, viaData, localOptimalityConstant,
                      acceptionFactor, rejectionFactor]
        
        testing_no_joint_reject = "joint_reject" in testing_args
        testing_no_length_lookups = "reuse_queries" in testing_args
        
        with ProcessPoolExecutor_ext(cpu_count, const_args) as pool:
            mapObj = pool.map(FlowPointGraph._find_admissible_via_vertices,
                              viaCandidates[disorder],
                              disorder,
                              repeat(testing_no_joint_reject),
                              repeat(testing_no_length_lookups),
                              tasklength=taskLength,
                              chunksize=chunksize)
            for num in mapObj:
                viaIndex, sourceIndices, sinkIndices, pathLengthsTmp, \
                                    res, notLO = num
                
                percentageDone = printCounter.next()
                if percentageDone:
                    self.prst(percentageDone, percent=True)
                countVia += res
                countNotLO += notLO
                viaVertexCount[sourceIndices, sinkIndices] += 1
                
                if testing:
                    pathLengths.extend(pathLengthsTmp)
                    continue
                
                sourceInspections = {source:
                                     FlowPointGraph._find_inspection_spots(
                                         viaIndex, #labelData[source][viaIndex]["parent"], 
                                         labelData[source], 
                                         inspectionArr)
                                     for source in np.unique(sourceIndices)}
                sinkInspections = {sink:FlowPointGraph._find_inspection_spots(
                                         viaIndex, labelData[sink+sourceNumber], 
                                         inspectionArr)
                                     for sink in np.unique(sinkIndices)}
                
                for sourceIndex, sinkIndex, pathLength in zip(sourceIndices, 
                                                              sinkIndices, 
                                                              pathLengthsTmp):
                    if not pathLengths[sourceIndex, sinkIndex]:
                        viaVertices[sourceIndex, sinkIndex] = [viaIndex]
                        pathLengths[sourceIndex, sinkIndex] = [pathLength]
                        pathIndex = 0
                    else: 
                        pathIndex = len(pathLengths[sourceIndex, sinkIndex])
                        viaVertices[sourceIndex, sinkIndex].append(viaIndex)
                        pathLengths[sourceIndex, sinkIndex].append(pathLength) 
                    
                    stations = sourceInspections[sourceIndex].union(
                                                    sinkInspections[sinkIndex])
                    for inspectionIndex in stations:
                        inspectedRoutes[inspectionIndex][
                                (sourceIndex, sinkIndex)].append(pathIndex)
                    stationCombinations[frozenset(stations)].append(
                                            (sourceIndex, sinkIndex, pathIndex)
                                                                    )
        
        self.decrease_print_level()
        startTime = endTime
        endTime = time.time()
        self.prst("Checking the admissibility of the candidate vertices took {} seconds.".format(round(endTime-startTime, 2)))
        self.prst("In total, {} seconds elapsed.".format(round(endTime-startTimeTot, 2)))
        
        if testing:
            testResults["result"]["mean paths"] = np.mean(viaVertexCount)
            testResults["result"]["median paths"] = np.median(viaVertexCount)
            for thrsh in range(1, 31):
                testResults["result"]["> {} paths".format(thrsh)] = np.mean(viaVertexCount > thrsh)
            testResults["result"]["pct 0.01"] = np.percentile(viaVertexCount, 1)
            testResults["result"]["pct 0.02"] = np.percentile(viaVertexCount, 2)
            testResults["result"]["pct 0.03"] = np.percentile(viaVertexCount, 3)
            testResults["result"]["pct 0.04"] = np.percentile(viaVertexCount, 4)
            testResults["result"]["pct 0.05"] = np.percentile(viaVertexCount, 5)
            testResults["result"]["pct 0.10"] = np.percentile(viaVertexCount, 10)
            testResults["result"]["pct 0.30"] = np.percentile(viaVertexCount, 30)
            testResults["result"]["pct 0.70"] = np.percentile(viaVertexCount, 70)
            testResults["result"]["pct 0.90"] = np.percentile(viaVertexCount, 90)
            testResults["result"]["pct 0.95"] = np.percentile(viaVertexCount, 95)
            testResults["result"]["pct 0.96"] = np.percentile(viaVertexCount, 96)
            testResults["result"]["pct 0.97"] = np.percentile(viaVertexCount, 97)
            testResults["result"]["pct 0.98"] = np.percentile(viaVertexCount, 98)
            testResults["result"]["pct 0.99"] = np.percentile(viaVertexCount, 99)
            testResults["result"]["mean shortest dists"] = np.mean(dists)
            testResults["result"]["median shortest dists"] = np.median(dists)
            testResults["result"]["mean dists"] = np.mean(pathLengths)
            testResults["result"]["median dists"] = np.median(pathLengths)
            testResults["time"]["admissibility"] = endTime-startTime
            testResults["time"]["total"] = endTime-startTimeTot
            testResults["time"]["path"] = testResults["time"]["total"] / testResults["result"]["mean paths"]
            testResults["time"]["slowdown"] = testResults["time"]["total"] / testResults["time"]["shortest path"]
        
            self.decrease_print_level()
            return testResults
        
        
        # because we excluded some via edges, it can theoretically happen that
        # some pairs have no via vertex at all. This should not be the case - 
        # at least the shortest path should always be included. Therefore, we
        # have to consider this case separately, if necessary.
        if not pathLengths.all():
            dSources, dSinks = np.nonzero(~pathLengths.astype(bool))
            warnings.warn("Not all pairs are connected. " + str(len(dSources)))
            for so, si in zip(dSources, dSinks):
                print("Disconnected:", so, self.sourceIndexToSourceID[so],
                      si, self.sinkIndexToSinkID[si], dists[so, si])
            """
            self.prst("Adding additional via vertex for not yet connected",
                      "pairs.")
            raise NotImplementedError("Finding the inspection paths on a " +
                                      "shortest route has not been implemented",
                                      + " yet because it seemed not to be "
                                      + " necessary. Obviously this is wrong..."
                                      )
            for source, sink in zip(np.nonzero(~pathLengths.astype(bool))):
                pathLengths[source, sink] = [dists[source, sink]]
                # TODO
                # here I have to implement an inspection spot finder for the
                # shortest path
            """
        #pathLengths[0,0]=0
        pathLengths = csr_matrix_nd(pathLengths)
        
        
        
        self.prst(countVia/(sourceNumber*sinkNumber), "via vertices per pair")     
        self.prst(countNotLO/(sourceNumber*sinkNumber), "via vertices per pair were pruned because they were not locally optimal")     
        minCount = np.min(viaVertexCount)
        maxCount = np.max(viaVertexCount)
        #bins=maxCount-minCount
        #if bins:
        #    print(np.histogram(viaVertexCount, bins=bins))     
        
        """
        def getVertexByID(ID, subset=None):
            if subset is None:
                return np.nonzero(self.vertices.array["ID"]==ID)[0][0]
            return np.nonzero(self.vertices.array["ID"][subset]==ID)[0][0]
        """
        
        self.decrease_print_level() 
        
        self.prst("done.")       
        
        #print(inspectedRoutes)
        
        # pathLengths is a 3D array that contains for each source-sink pair a
        # list (as a  sparse matrix) of the lengths of all paths that go from 
        # the source to the sink
        #
        # inspectedRoutes is a dict that contains for each inspection station 
        # (key) another dict with key source-sink pair and as value a list of 
        # the indices of all paths that go through the inspection spot from the 
        # source to the sink. The paths's index refers to its index in 
        # pathLengths. That is, the lengths of the paths going from i to j 
        # through spot s are given by 
        # [pathLengths[i, j, p] for p in inspectedRoutes[s][i, j]]
        #
        # stationCombinations is a dictionary that contains frozen sets of 
        # inspection stations as key and the flows that go exactly via these
        # sets of inspection stations as values - in form of a list of tuples
        # (sourceIndex, sinkIndex, flowIndex) (see above)
        
        inspectedRoutes = dict(inspectedRoutes)
        for key, value in inspectedRoutes.items():
            inspectedRoutes[key] = dict(value)
        
        return pathLengths, inspectedRoutes, dict(stationCombinations)
                
    @staticmethod
    def _decrease_closest_source_distance(queue, closestSourceDists):
        while True:
            data = queue.get()
            if data is not None:
                for vertexIndex, dist in data:
                    if closestSourceDists[vertexIndex] > dist:
                        closestSourceDists[vertexIndex] = dist
            else:
                return closestSourceDists
    
    @staticmethod
    def _grow_bounded_tree(vertexArr, lengthArr, inspectionArr,
                           stretchConstant, localOptimalityConstant, 
                           closestSourceDistCommunicator, #treeIndex, 
                           startIndex, forward, shortestShortestDist,
                           longestShortestDist, naiveBound=False,
                           naivePruning=False, noBackwardPruning=False,
                           ):
        
        edgesVisited = set()
        #tolerance = 1e-11
        rTol = 1+1e-7
        
        chunk = []
        chunksize = 1000
        if forward: 
            closestSourceQueue = closestSourceDistCommunicator
        else:
            closestSourceDists = closestSourceDistCommunicator
        
        initSize = 25000
        reachArr = vertexArr["reachBound"] 
        if forward: 
            neighborArr = vertexArr["successors"]
        else:
            neighborArr = vertexArr["predecessors"]
        
        queue = intquickheapdict(((startIndex, 0),), initSize=initSize)
        
        dataDtype = {"names":["parent", "edge", "cost", "parent_inspection"],
                     "formats":["int", "int", "double", "int"]} 
        
        data = FlexibleArrayDict(initSize, dtype=dataDtype) 
        data[startIndex] = (-1, -1, 0, -1)
        
        if naiveBound:
            bound = longestShortestDist * stretchConstant
        else:
            bound = (longestShortestDist * stretchConstant 
                     * max(1-localOptimalityConstant, 0.5))
            
        while queue: 
            thisVertex, thisCost = queue.popitem()
            
            if naivePruning:
                # for testing only
                pruned = (reachArr[thisVertex]*rTol < 
                          localOptimalityConstant/2 * thisCost)
            else:
                pruned = (reachArr[thisVertex]*rTol < 
                          min(thisCost, localOptimalityConstant/2 
                              * max(thisCost, shortestShortestDist)))
            
            # set the vertex information
            # However, we need only one overlapping vertex from the vertex 
            # perspective. From the edge perspective we do need one overlapping
            # edge, too, but not both its end points.
            # therefore, we can save memory here by deleting the unnecessary
            # end point.
            edge = data[thisVertex]["edge"]
            if edge >= 0: edgesVisited.add(edge)
            """ # uncomment this, if some vertices are not allowed to be 
                # via vertices
            parent, edge, _, _ = data[thisVertex]
            if forward: 
                if edge >= 0 and potentialViaVertexArr[thisVertex]:
                    edgesVisited[edge] = True
            else:
                if edge >= 0 and potentialViaVertexArr[parent]:
                    edgesVisited[edge] = True
            #vertexDists[thisVertex, treeIndex] = thisCost
            """
            
            if not forward and pruned:
                del data[thisVertex]
                continue
            else:
                data[thisVertex]["cost"] = thisCost
                
                if forward: 
                    chunk.append((thisVertex, thisCost))
                    if len(chunk) > chunksize:
                        closestSourceQueue.put(chunk)
                        chunk = []
                
                # prune, if necessary
                if pruned:
                    continue
            
            if inspectionArr[edge] and edge >= 0:
                inspection = thisVertex
            else:
                inspection = data[thisVertex]["parent_inspection"]
            
            # the edge to this vertex must be considered (unless pruned)    
            # even if the vertex is farther away than the bound. However,
            # it does not need to be expanded.
            if thisCost > bound:
                """
                if DEBUG2:
                    print("thisCost > bound")
                """
                continue
            
                
            # process successors
            for neighbor, edge in neighborArr[thisVertex].items():
                #print("neighbor")
                newCost = thisCost + lengthArr[edge]
                
                # early pruning only from one side so that
                # paths are always closed. 
                # Furthermore, there are no 
                # lower distance bounds in forward direction
                if not forward:
                    if naivePruning or noBackwardPruning:
                        pruned = False
                    else:
                        pruned = (reachArr[neighbor]*rTol <
                                  min(newCost, localOptimalityConstant/2 
                                      * max(newCost, shortestShortestDist),
                                      closestSourceDists[neighbor]))
                    
                    if pruned:
                        #if reach < thisCost: 
                        #    break    # ! This must be deleted, because the neighbors have different distances to the sources.
                        #elif reach < newCost:
                        """
                        if DEBUG2:
                            print("reach < newCost and reach < closestSourceDists[neighbor]",
                                  vertexArr[neighbor]["ID"])
                        """
                        continue
                
                # if not pruned
                neighborCost = queue.get(neighbor, -1.)
                
                if neighborCost >= 0:   # if neighbor is in the queue
                    update = neighborCost > newCost*rTol
                else:
                    #check whether neighbor already scanned
                    update = neighbor not in data
                
                """
                if DEBUG2:
                    print("Update", vertexArr[neighbor]["ID"], update, 
                          neighborCost, newCost + tolerance, neighborCost-(newCost + tolerance))
                """
                
                # reach < thisCost is the weakened early pruning #?????
                if update: # and reach >= thisCost: ????
                    data[neighbor] = (thisVertex, edge, np.inf, inspection)
                    queue[neighbor] = newCost
            
        if forward:
            closestSourceQueue.put(chunk)
        data.cut()
        return data, edgesVisited 
        """
        j = 0
        for i in range(10000000000):
            j = i
        print("done2")
        return 0, 0
    #"""
    @staticmethod
    def _find_edge_superneighbours(edgeVisitedSources, edgeVisitedSinks,
                                   edgeIndex, predecessorEdges, successorEdges):
        
        """
        If we disregard empty sets, we know that each edge has at most
        one superset or subset for each predecessor or successor.
        Here we determine the respective neighbour edges. If there is
        a superset neighbour, we know that the respective edge is no plateau
        peak. Hence, _find_edge_superneighbours returns None.
        Otherwise, _find_edge_superneighbours returns the indices of the edges
        (predecessor, successor) that have the same set of sources and sinks.
        If there is no such predecessor or successore, the entry is None
        respectively. For example, direct plateau peaks have only strict subsets
        and will return (None, None).
        """
        
        sources, sinks = edgeVisitedSources[edgeIndex], edgeVisitedSinks[edgeIndex]
        
        for neighborEdgeIndex in predecessorEdges.values():
            # recall that each edge in edgeVisitedSinks is also in 
            # edgeVisitedSources by construction
            if neighborEdgeIndex not in edgeVisitedSources:
                continue
            neighborSources = edgeVisitedSources[neighborEdgeIndex]
            neighborSinks = edgeVisitedSinks[neighborEdgeIndex]
            if sources.issubset(neighborSources) and sinks.issubset(neighborSinks):
                if neighborSources == sources and neighborSinks == sinks:
                    predecessor = neighborEdgeIndex
                    break
                else:
                    return None
        else:
            predecessor = None
        
        for neighborEdgeIndex in successorEdges.values():
            # recall that each edge in edgeVisitedSinks is also in 
            # edgeVisitedSources by construction
            if neighborEdgeIndex not in edgeVisitedSources:
                continue
            neighborSources = edgeVisitedSources[neighborEdgeIndex]
            neighborSinks = edgeVisitedSinks[neighborEdgeIndex]
            if sources.issubset(neighborSources) and sinks.issubset(neighborSinks):
                if neighborSources == sources and neighborSinks == sinks:
                    successor = neighborEdgeIndex
                    break
                else:
                    return None
        else:
            successor = None
        
        return predecessor, successor
            
    def _find_plateau_peaks(self, candidateEdges, edgesVisitedSources,
                            edgesVisitedSinks):
        
        successorArr = self.vertices.array["successors"] 
        predecessorArr = self.vertices.array["predecessors"] 
        
        vertexFromIndices = self.edges.array["fromIndex"]
        vertexToIndices = self.edges.array["toIndex"]
        
        
        plateauPeaks = []
        unprocessedCandidates = {}
        
        self.prst("Searching for plateau peak edges. ({} candidates)".format(
                                                           len(candidateEdges)))
        
        self.increase_print_level()
        
        taskLength = len(edgesVisitedSinks)
        const_args = (edgesVisitedSources, edgesVisitedSinks)
        
        printCounter = Counter(len(candidateEdges), 0.01)
        

        with ProcessPoolExecutor_ext(None, const_args) as pool:
            mapObj = pool.map(FlowPointGraph._find_edge_superneighbours,
                              candidateEdges,
                              predecessorArr[vertexFromIndices[candidateEdges]], 
                              successorArr[vertexToIndices[candidateEdges]], 
                              tasklength=taskLength)
            for edgeIndex, neighborTuple in zip(candidateEdges, mapObj):
                percentageDone = printCounter.next()
                if percentageDone:
                    self.prst(percentageDone, percent=True)
                if neighborTuple is None:
                    # this edge has a superset neighbour
                    continue
                elif neighborTuple == (None, None):
                    # this edge is known to be a plateau peak
                    plateauPeaks.append(edgeIndex)
                else: 
                    unprocessedCandidates[edgeIndex] = neighborTuple
        
        self.prst("Traversing the graph.")
        noNeighbour = (False, False)
        while unprocessedCandidates:
            edgeIndex, neighborTuple = unprocessedCandidates.popitem()
            predecessorIndex, successorIndex = neighborTuple
            
            # traverse predecessors and successors of plateaus
            # until either no superset is found (index is None) or there is an 
            # index that has already been discarded because it is known to be 
            # no plateau peak (then the entry is missing). 
            while predecessorIndex or (predecessorIndex is not None and
                                       predecessorIndex is not False):
                predecessorIndex = unprocessedCandidates.pop(predecessorIndex, 
                                                             noNeighbour)[0]
            if predecessorIndex is not None:
                continue
            while successorIndex or (successorIndex is not None and
                                     successorIndex is not False):
                successorIndex = unprocessedCandidates.pop(successorIndex, 
                                                           noNeighbour)[1]
            if successorIndex is None:
                plateauPeaks.append(edgeIndex)
                
            
        self.decrease_print_level()
        self.prst("Found", len(plateauPeaks) ,"plateau peak edges.")
        
        return np.array(plateauPeaks)

        
        
                
    """
    @staticmethod
    def _get_edge_source_sink_distances(fbLabelData, edgesVisitedArray, 
                                        vertexIndex):
        
        return [(labelData[vertexIndex]["cost"] if visited else np.nan)
                for labelData, visited in zip(fbLabelData, edgesVisitedArray)]
    """
    
    @staticmethod
    def _get_edge_source_sink_distances(fbLabelData, edgeVisitedSources, 
                                        edgeVisitedSinks, 
                                        vertexIndex, sourceNo, sinkNo):
        resultSources = np.full(sourceNo, np.nan)
        resultSinks = np.full(sinkNo, np.nan)
        for s in edgeVisitedSources:
            resultSources[s] = fbLabelData[s][vertexIndex]["cost"]
        for s in edgeVisitedSinks:
            resultSinks[s] = fbLabelData[s+sourceNo][vertexIndex]["cost"]
        return resultSources, resultSinks
    
    @staticmethod
    def _get_vertex_source_sink_distances(fbLabelData, vertexIndex):
        
        return [(labelData.array[labelData.indexDict[vertexIndex]]["cost"]
                 if vertexIndex in labelData.indexDict else np.nan)
                for labelData in fbLabelData]
    
    
    @staticmethod
    def _find_vertexCandidates(shortestDistances, sourceDistances, sinkDistances, 
                               stretchConstant, candidates, pair, unique=True):
        roundNo = 8
        sourceIndex, sinkIndex = pair
        viaDistances = sourceDistances[:,sourceIndex]+sinkDistances[:,sinkIndex]
        with np.errstate(invalid='ignore'):
            viaDistanceIndices = np.nonzero(viaDistances <= 
                shortestDistances[sourceIndex, sinkIndex]*stretchConstant)[0]
        lengths = viaDistances[viaDistanceIndices]
        
        # only for testing purposes we may want to return the same path multiple 
        # times
        if not unique:
            return candidates[viaDistanceIndices], lengths
            
        lengths, candidateIndices = unique_tol(
            round_rel(lengths, roundNo), 
            round_rel(lengths, roundNo, True),
            lengths
            )
        
        return candidates[viaDistanceIndices[candidateIndices]], lengths
        
    
    
    
    @staticmethod
    def _find_admissible_via_vertices(
            vertexArr, edgeArr, shortestDistances, labelData, viaData,
            localOptimalityConstant, acceptionFactor, rejectionFactor,
            vertexIndex, viaIndex, testing_no_joint_reject=False,
            testing_no_length_lookups=False):
        """
        vertexArr
                    Structured Array with the vertex information. Indexed by 
                    vertexIndex
        edgeArr
                    Structured Array with the edge information. Indexed by 
                    edgeIndex
        shortestDistances
                    2D double Array; has at [i,j] the distance from source i to 
                    sink j
                    i,j:    are the index of the source/sink in their 
                            respective lists and NOT the vertexIndices of the 
                            respective vertices!
        labelData
                    list of FlexibleArrayDicts 
                    with fields ["parent", "edge", "cost"];
                    has at [i][j] the distance of vertex j to source/sink i
                    i:      number of the via vertex, NOT its vertex index
                    j:      index of the source/sink in the source/sink list. 
                            NOT the vertex index!
                            If j is a sink, then the index in the sink list is
                            given by j-SourceNumber
        localOptimalityConstant
                    search parameter
        acceptionFactor, rejectionFactor
                    determie the precision of the results. Paths with local 
                    optimality above localOptimalityConstant*acceptFactor might 
                    be accepted. Local optimal paths with local optimality below
                    localOptimalityConstant*rejectFactor might be rejected.
                    Perfect results are obtained with the values being (1, 1)
                    A 2-approximation can be obtained with the values (2/3, 4/3)
        candidateIndex
                    number of the via vertex (NOT its vertex index) that is to
                    be checked for admissibility w.r.t. all source/sink pairs
        vertexIndex
                    vertex index of the vertex that is to
                    be checked for admissibility w.r.t. all source/sink pairs
        """
        rTol = 1e-6
        rTolFact = 1+rTol
        
        #========= Performing T-Tests ==========================================
        
        # save the pair data in a convenient way
        pairList = viaData[viaIndex]
            
        sourceNumber, sinkNumber = shortestDistances.shape
        
        # get the original indices of the sources and sinks
        consideredSourceIndices = np.unique(pairList["source"])
        # for the sinks we add the sourceNumber, so that we reach the correct
        # entry in the labelData
        consideredSinkIndices_plain = np.unique(pairList["sink"])
        consideredSinkIndices = consideredSinkIndices_plain + sourceNumber
        
        consideredSources = np.arange(consideredSourceIndices.size)
        consideredSinks = np.arange(consideredSinkIndices.size)
        
        
        sourceIndexToSource = np.zeros(sourceNumber, dtype=int)
        sourceIndexToSource[consideredSourceIndices] = consideredSources
        sinkIndexToSink = np.zeros(sinkNumber, dtype=int)
        sinkIndexToSink[consideredSinkIndices_plain] = consideredSinks
        
        shortestDistances = shortestDistances[consideredSourceIndices][:,consideredSinkIndices_plain]
        
        # replace source and sink indices in pairList with internal indices
        pairList["source"] = sourceIndexToSource[pairList["source"]]
        pairList["sink"] = sinkIndexToSink[pairList["sink"]]
        
        # distances[i,j] contains the distance of the via path from source i
        # to sink j
        distances = np.ma.array(np.zeros((consideredSources.size, 
                                          consideredSinks.size)), mask=True)
        
        
        distances[pairList["source"], pairList["sink"]] = pairList["distance"]
        
        if np.isnan(pairList["distance"]).any():
            print("ISNAN0", pairList["distance"], pairList)
        
        admissiblePairs = ~distances.mask
        
        sourcePairIndices, sinkPairIndices = np.meshgrid(
                                                np.arange(distances.shape[0]),
                                                np.arange(distances.shape[1]),
                                                indexing='ij'
                                                )
        pairList.sort(order="distance")
        
        
        vertexVisitedFromSource = FlexibleArrayDict((1000, 
                                                     distances.shape[0]),
                                                     dtype=bool)
        vertexVisitedFromSink = FlexibleArrayDict((1000, distances.shape[1]), 
                                                   dtype=bool)
        
        # note for each sink the maximal distance of a pair involving it
        maxPairDistSinks = np.max(distances, 0)
        maxPairDistSinks = maxPairDistSinks[~maxPairDistSinks.mask]
        # do the same for each source
        maxPairDistSources = np.max(distances, 1)
        maxPairDistSources = maxPairDistSources[~maxPairDistSources.mask]
        
        reverseSourceLabelData = {}
 
        # for each source and sink, mark all vertices visited on the way to
        # the respective end point as visited
        for endPoints, endPointIndices, maxPairDist, visitedArr, rSLD in (
                (consideredSources, consideredSourceIndices, 
                 maxPairDistSources, vertexVisitedFromSource,
                 reverseSourceLabelData),
                (consideredSinks, consideredSinkIndices, 
                 maxPairDistSinks, vertexVisitedFromSink, None),
                                                       ):
            #visitedArr[vertexIndex] = False
            for endPoint, maxDist in zip(endPoints, maxPairDist):
                thisLabelData = labelData[endPointIndices[endPoint]]
                maxDist = localOptimalityConstant * maxDist
                thisVertex = vertexIndex
                parent, _, cost, _ = thisLabelData[thisVertex]
                stopCost = cost - maxDist
                #visitedArr[thisVertex][endPoint] = True
                
                if rSLD is not None:
                    reverseData = {}
                    rSLD[endPoint] = reverseData
                while cost >= stopCost and thisVertex >= 0:
                    parent, _, cost, _ = thisLabelData[thisVertex]
                    if thisVertex not in visitedArr:
                        visitedArr[thisVertex] = False
                    visitedArr[thisVertex][endPoint] = True
                    
                    if rSLD is not None and parent >= 0:
                        reverseData[parent] = (thisVertex, cost)
                    thisVertex = parent
        
        sourcePointers = defaultdict(lambda: vertexIndex)
        
        # Dict that contains all the pairs for which we have confirmed that 
        # the subpath between them is locally optimal 
        # Key: test vertex on the sink branch
        # Value: set(tuple(test partner on source branch, 
        #                    its predecessor (w.r.t. v)))
        successfullyTested = defaultdict(lambda: set())
        
        resultSourceIndices = []
        resultSinkIndices = []
        resultLengths = []
        
                           
        
        # now check all pairs for their admissibility
        sourceSinkIndex = -1
        sourceList = pairList["source"]
        sinkList = pairList["sink"]
        while True:
            # if we have already found out that we do not have to consider the
            # pair (anymore), continue
            sourceSinkIndex = find_next_nonzero2d(admissiblePairs, sourceList,
                                                  sinkList, sourceSinkIndex+1)
            
            if sourceSinkIndex is None:
                break
            
            source, sink, dist = pairList[sourceSinkIndex]
            
            sourceLabelData = labelData[consideredSourceIndices[source]]
            sinkLabelData = labelData[consideredSinkIndices[sink]]
            thisReverseSourceLabelData = reverseSourceLabelData[source]
            
            # T is the local optimal length
            T = localOptimalityConstant * dist
            
            extT = rejectionFactor * T
            
            
            # baseCostSink is the cost of the via vertex from the sink 
            parent, _, baseCostSink, _ = sinkLabelData[vertexIndex]
            _, _, baseCostSource, _ = sourceLabelData[vertexIndex]
            
            # if the via vertex is an end vertex of the total path,
            # it is locally optimal, if the via cost is equal the shortest cost
            # (note that the via vertex is just the end point of the actual      NO A via edge must be scanned from both direction. Hence, only the shortest path will be used.
            # via edge - therefore, the via cost can be higher than the actual
            # cost to the end point) or not locally optimal (because we are 
            # guaranteed that pruning does not lead to wrong results for 
            # sufficiently locally optimal paths. However, we obtained that the
            # end vertex will be reached with too high cost, which is a wring 
            # result. Therefore, the via path cannot be locally optimal.
            # We only have to compare the costs
            if baseCostSink <= rTol or baseCostSource <= rTol:
                #if np.isclose(dist, shortestDistances[source, sink], rTol):
                resultSourceIndices.append(source)
                resultSinkIndices.append(sink)
                resultLengths.append(dist)
                
                continue
            
            
            # go to the x_T vertex (farthest vertex within the test range)
            # (this could be sped up, if necessary)
            thisSourceVertex = sourcePointers[source]
            sourceParent, _, sourceCost, _ = sourceLabelData[thisSourceVertex]
            sourceParentTmp, _, sourceCostParent, _ = sourceLabelData[
                                                                 sourceParent]
            stopCost = baseCostSource - T
            
            # note: I want to know the farthest vertex in T-Range and then
            # use its parent for the test
            while sourceCostParent >= stopCost and sourceParentTmp >= 0:
                thisSourceVertex = sourceParent
                sourceParent = sourceParentTmp
                sourceCost = sourceCostParent
                sourceParentTmp, _, sourceCostParent, _ = sourceLabelData[
                                                                sourceParent]
            
            # Note the source branch end of the considered subpath
            sourceSubpathBound = sourceParent
            sinkSubpathBound = -1
            
            # Due to our skewed graph we have some very long edges. We do not
            # want them to interfere with the local optimality checks.
            if sourceCost - sourceCostParent >= 2*T:
                sourceParent = thisSourceVertex
                #thisSourceVertex = thisSourceChield
                sourceCostParent = sourceCost
            
            
            # update the source pointer
            sourcePointers[source] = thisSourceVertex
            
            # Do the T-Tests
            sinkCost = baseCostSink
            started = True  # necessary in case the first check vertex is a 
                            # direct neighbor of the via vertex
            thisSinkVertex = vertexIndex
            sinkParent, _, sinkCost, _ = sinkLabelData[thisSinkVertex]
            
            
            while sourceCostParent < baseCostSource and sinkSubpathBound < 0:
                
                # find the partner vertex on the sink branch
                sinkCostOld = sinkCost
                
                # Possible optimization: if stopCost <= 0, we could set the
                # sinkParent to the sink right away and save some interations.
                
                # we do not have to go further than T away from the via 
                # vertex
                stopCost = max(baseCostSink + baseCostSource 
                               - sourceCost - extT, baseCostSink - T)
                # needed because we can use previous results only if the 
                # distance is sufficiently large
                minStopCost = max(baseCostSink + baseCostSource 
                                  - sourceCost - T, baseCostSink - T)
                sinkParentTmp, _, sinkCostParent, _ = \
                                                sinkLabelData[sinkParent]
                # we might be able to exploit earlier tests
                sourceParentTested = successfullyTested.get(sourceParent, False)
                alreadyTested = False
                
                if sinkCostParent >= stopCost and sinkParentTmp >= 0:
                    while sinkCostParent >= stopCost and sinkParentTmp >= 0:
                        thisSinkVertex = sinkParent
                        sinkParent = sinkParentTmp
                        sinkCost = sinkCostParent
                        sinkParentTmp, _, sinkCostParent, _ = sinkLabelData[
                                                                    sinkParent]
                        
                        
                        # if we know that the path between sourceParent and 
                        # sinkParent is a shortest paht, then we can skip a shortest
                        # path query
                        if (sinkCostParent <= minStopCost and sourceParentTested 
                                and sinkParent in sourceParentTested):
                            alreadyTested = True
                            break
                elif sourceParentTested and sinkParent in sourceParentTested:
                    alreadyTested = True
                
                # Again: Due to our skewed graph we have some very long  
                # edges. We do not want them to interfere with the local 
                # optimality checks.
                if sinkCost - sinkCostParent >= 2*T:
                    sinkSubpathBound = sinkParent
                    sinkParent = thisSinkVertex
                    sinkCostParent = sinkCost
                
                # if a step has been done, 
                if not alreadyTested and (not sinkCostOld == sinkCost 
                                          or started):
                    started = False
                    
                    compCost = (baseCostSink + baseCostSource 
                                - sinkCostParent - sourceCostParent)
                        
                    # if considered section is shortest path
                    #if (localDistBound + aTol >= compCost or 
                    if (find_shortest_distance(vertexArr, edgeArr, 
                                               sourceParent, sinkParent)
                         * rTolFact >= compCost):
                        #if localDist + aTol >= compCost:
                        if not testing_no_length_lookups:
                            successfullyTested[sourceParent].add(sinkParent)
                    else:
                        if testing_no_joint_reject:
                            admissiblePairs[source, sink] = False
                            break
                        # note all vertices with via paths over 
                        # this subsection as not considered
                        admissiblePairs[np.ix_(
                                vertexVisitedFromSource[sourceParent],
                                vertexVisitedFromSink[sinkParent],
                                )] = False
                        break
                
                
                # update the vertex on the source branch
                stopCost = baseCostSink + baseCostSource - sinkCost - T
                while True:
                    sourceParent = thisSourceVertex
                    sourceCostParent = sourceCost
                    try:
                        thisSourceVertex, sourceCost = \
                                thisReverseSourceLabelData[thisSourceVertex]
                    except KeyError:
                        assert sourceCostParent >= baseCostSource
                        break
                    if sourceCost > stopCost:
                        break
            # if all tests were sucessful
            else:    
                # only for testing purposes: continue without accepting
                # further origin-destination pairs
                if testing_no_joint_reject:
                    resultSourceIndices.append(source)
                    resultSinkIndices.append(sink)
                    resultLengths.append(dist)
                    continue
                
                # if we have not stopped the search artificially because of a 
                # too long edge
                if sinkSubpathBound < 0:
                    sinkSubpathBound = sinkParent
                
                # we consider all not yet processed pairs for which local 
                # optimality for the tested path is sufficient (taking into 
                # account the acception factor)
                considered = np.ix_(vertexVisitedFromSource[sourceSubpathBound],
                                    vertexVisitedFromSink[sinkSubpathBound])
                considered2 = np.logical_and(admissiblePairs[considered],
                                             distances[considered] 
                                             <= T/(acceptionFactor*
                                                   localOptimalityConstant)*rTolFact)
                
                sourceProcessIndices = sourcePairIndices[considered][considered2]
                sinkProcessIndices = sinkPairIndices[considered][considered2]
                
                # mark the respective pairs as processed
                admissiblePairs[sourceProcessIndices, 
                                sinkProcessIndices] = False
                # note the results
                resultSourceIndices.extend(sourceProcessIndices)
                resultSinkIndices.extend(sinkProcessIndices)
                resultLengths.extend(distances[considered][considered2])
        
        ########### Be careful with machine imprecision!
        
        # return result
        admissiblePairNumber = len(resultSourceIndices)
        notLO = len(pairList) - admissiblePairNumber
        
        return (vertexIndex, consideredSourceIndices[resultSourceIndices], 
                consideredSinkIndices_plain[resultSinkIndices], resultLengths, 
                admissiblePairNumber, notLO) 

    
    @staticmethod
    def _find_inspection_spots(start, labelData, inspectionArr):
        if start < 0 or labelData[start]["edge"] < 0: 
            return set()
        if inspectionArr[labelData[start]["edge"]]:
            spots = cp.copy(inspectionArr[labelData[start]["edge"]])
        else:
            spots = set()
        thisVertex = labelData[start]["parent_inspection"]
        while thisVertex >= 0:
            #print("thisVertex", thisVertex)
            #print('labelData[thisVertex]["edge"]', labelData[thisVertex]["edge"])
            #print("spots", inspectionArr[labelData[thisVertex]["edge"]])
            spots.update(inspectionArr[labelData[thisVertex]["edge"]])
            thisVertex = labelData[thisVertex]["parent_inspection"]
        return spots


if __name__ == "__main__":
    import traceback
    import timeit
    iDType = "|S10"
    #"""
    fileNameEdges = "LakeNetworkExample_full.csv"
    fileNameVertices = "LakeNetworkExample_full_vertices.csv"
    
    #fileNameEdges = "LakeNetworkExample_small.csv"
    #fileNameVertices = "LakeNetworkExample_small_vertices.csv"
    """
    fileNameEdges = "ExportEdges.csv"
    fileNameVertices = "ExportVertices2.csv"
    #fileNameEdges = "ExportEdges_North.csv"
    #fileNameVertices = "ExportVertices2.csv"
    pairList = ((b'231421', b'768396'),
                (b'J54131', b'768396'),
                (b'J54175', b'670659'), 
                (b'J54153', b'998327'), 
                (b'J54163', b'463830'), 
                (b'J54185', b'91769'))
                
    #pairList = ((b'231421', b'768396'),)
    """
    print("Starting test. (29)")
    print("Reading files.")
    
    edges = np.genfromtxt(fileNameEdges, delimiter=",",  
                                   skip_header = True, 
                                   dtype = {"names":["ID", "from_to_original", "cost", 
                                                     "inspection", "lakeID"], 
                                            'formats':[iDType, '2' + iDType, "double", 
                                                       "3" + iDType, iDType]},
                              autostrip = True)
    vertexData = np.genfromtxt(fileNameVertices, delimiter=",",  
                                   skip_header = True, 
                                   dtype = {"names":["ID", "type", "infested"], 
                                            'formats':[iDType, 'int', 'bool']},
                                   autostrip = True)
    
    #TODO: make inspection such that each edge has only one inspection flag!
    ft = np.vstack((edges["from_to_original"][:,::-1], edges["from_to_original"]))
    edata = np.concatenate((edges[["ID", "cost", "inspection", "lakeID"]],
                            edges[["ID", "cost", "inspection", "lakeID"]]))
    
    print("Creating flexible graph.")
    g = FlexibleGraph(ft, edata, vertexData["ID"], vertexData[["type", "infested"]],
                      replacementMode="shortest", lengthLabel="cost")
    g.set_default_vertex_data((0, 0, False))
    
    """
    print(g.get_edge_data(b'2', b'8', False))
    print(g.edges.get_array()[["fromID", "toID"]])
    print(g.edges.get_array().size)
    print(g.get_neighbor_edges(b'2', copy=True))
    """
    
    g.add_vertex_attributes("significant", bool, g.vertices.array.type > 0)
    #g.add_vertex_attributes("significant", bool, (True,))
    
    
    print("Removing insignificant dead ends.")
    g.remove_insignificant_dead_ends("significant")
    #g.vertices.array["significant"][:] = True
    
    """
    print(g.edges.get_array()[["fromID", "toID"]])
    print(g.edges.get_array().size)
    print("--------------------------")
    """
    
    #print("Creating fast graph.")
    #g2 = FastGraph(g)
    
    """
    print(g2.edges.get_array()[["fromID", "toID"]])
    print(g2.vertices.array)
    g2.remove_vertex(0)
    print(g2.edges.get_array()[["fromID", "toID", "fromIndex", "toIndex"]])
    print(g2.vertices.get_array().ID)
    """
    """
    parameterList = [(1, 3, 3, (1.5, 1.5, 2.5)), 
                     (1, 3, 4, (1.5, 1.5, 2.5)),
                     (1, 3, 4, (1, 1, 1.5)),
                     (1, 3, 4, (2, 2, 3)),
                     (2, 2, 3, (1.5, 1.5, 2.5)),
                     (2, 3, 3, (1.5, 1.5, 2.5)),
                     (0.5, 2, 3, (1.5, 1.5, 2.5)),
                     (0.5, 3, 4, (1.5, 1.5, 2.5)),
                     (3, 2, 3, (1.5, 1.5, 2.5)),
                     (3, 3, 3, (1.5, 1.5, 2.5))]
    """
    parameterList=[(1, 3, 4, (2, 2, 3))]
    #"""
    
    g3 = FlowPointGraph(g, "cost", "significant")
    #g3.set_silent_status(True)
    profile("g3.preprocessing(1, 3, 4, expansionBounds=(2, 2, 3))", 
            globals(), locals())
    fromIndices = g3.vertices.get_array_indices("type", 1) 
    toIndices = g3.vertices.get_array_indices("type", 2)
    
    
    for parameters in parameterList:
        try:
            print("Creating FlowPointGraph.")
            
            print("Preprocessing...")
            
            profile("g3.find_alternative_paths(fromIndices, toIndices, 1.5, 0.2, 1)",
                    globals(), locals())
            
            """
            prepStr = "g3.preprocessing({}, {}, {}, expansionBounds={})".format(*parameters)
            print(prepStr)
            #profile("g3.preprocessing(2, 2, 3, expansionBounds=(1.5, 1.5, 2.5))", globals(), locals())
            print("full:", timeit.timeit(prepStr, number=1, globals=globals()))
            """
            #g3.preprocessing(1, 3, expansionBounds=(1, 1.5, 1.5, 2))
            
            #g3.vertices.array.significant = g3.vertices.array.type > 0 
            
        except Exception as e:
            traceback.print_exc()
    
    print("Done.")